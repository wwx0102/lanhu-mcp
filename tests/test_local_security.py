"""Regression coverage for local file boundaries and outbound credential scope."""

from pathlib import Path

import httpx
import pytest

import lanhu_mcp_server as server


@pytest.mark.parametrize("project_id", ["../../outside", "/tmp/outside", r"..\outside", "", ".", "x/y"])
def test_message_store_rejects_path_components(tmp_path, monkeypatch, project_id):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    with pytest.raises(ValueError, match="identifier"):
        server.MessageStore(project_id)
    assert not list(tmp_path.rglob("*.json"))


def test_message_store_preserves_valid_ids_and_rejects_escaping_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    store = server.MessageStore("project-123")
    store.record_collaborator("Local user", "Developer")
    assert store.file_path == tmp_path / "data/messages/project-123.json"
    outside = tmp_path / "outside.json"
    outside.write_text("untouched")
    (store.storage_dir / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        server.MessageStore("linked")
    assert outside.read_text() == "untouched"


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", r"..\outside", "C:/outside", "./", ""])
def test_resource_paths_reject_escapes(tmp_path, relative):
    with pytest.raises(ValueError):
        server._safe_resource_path(tmp_path, relative)


def test_resource_paths_reject_escaping_symlinks(tmp_path):
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        server._safe_resource_path(resource, "linked/outside.js")
    assert server._safe_resource_path(resource, "files/page.js") == resource / "files/page.js"


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("pid", "../../outside"), ("docId", "../../outside")])
async def test_url_ids_cannot_escape_cache_paths(field, value):
    extractor = server.LanhuExtractor()
    query = {"pid": "project", "docId": "document", field: value}
    try:
        with pytest.raises(ValueError, match="identifier"):
            extractor.parse_url("&".join(f"{key}={item}" for key, item in query.items()))
    finally:
        await extractor.close()


@pytest.mark.asyncio
async def test_resource_mapping_cannot_overwrite_outside_file(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    resource = tmp_path / "resource"
    resource.mkdir()
    extractor = server.LanhuExtractor()
    try:
        with pytest.raises(ValueError, match="cache directory"):
            await extractor._download_page_resources({
                "scripts": {"../outside.txt": {"sign_md5": "test.js"}},
            }, resource)
    finally:
        await extractor.close()
    assert outside.read_text() == "untouched"


def _mock_network(monkeypatch, handler):
    original = httpx.AsyncClient

    def client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(**kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", client)
    monkeypatch.setattr(server, "COOKIE", "session=synthetic-main")
    monkeypatch.setattr(server, "DDS_COOKIE", "session=synthetic-dds")


@pytest.mark.asyncio
@pytest.mark.parametrize("host,expected_cookie,expected_auth", [
    ("lanhuapp.com", "session=synthetic-main", None),
    ("dds.lanhuapp.com", "session=synthetic-dds", "Basic dW5kZWZpbmVkOg=="),
    ("axure-file.lanhuapp.com", None, None),
    ("lanhu.oss-cn-beijing.aliyuncs.com", None, None),
    ("lanhu-dds-backend.oss-cn-beijing.aliyuncs.com", None, None),
])
async def test_credentials_only_reach_their_origin(monkeypatch, host, expected_cookie, expected_auth):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=b"safe")

    _mock_network(monkeypatch, handler)
    extractor = server.LanhuExtractor()
    try:
        await extractor.client.get(f"https://{host}/test")
    finally:
        await extractor.close()
    assert len(requests) == 1
    assert requests[0].headers.get("Cookie") == expected_cookie
    assert requests[0].headers.get("Authorization") == expected_auth


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://attacker.invalid/file.js", "https://lanhuapp.com.attacker.invalid/file.js",
    "http://lanhuapp.com/file.js", "https://127.0.0.1/file.js",
    "https://lanhuapp.com:444/file.js", "https://user:password@lanhuapp.com/file.js",
])
async def test_unapproved_resource_origins_never_reach_transport(monkeypatch, url):
    def unexpected(request):
        raise AssertionError("Rejected URL reached the network")

    _mock_network(monkeypatch, unexpected)
    extractor = server.LanhuExtractor()
    try:
        with pytest.raises(ValueError, match="approved HTTPS"):
            await extractor.client.get(url)
    finally:
        await extractor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["https://axure-file.lanhuapp.com/file.js", "https://attacker.invalid/file.js"])
async def test_redirects_revalidate_origin_and_strip_credentials(monkeypatch, destination):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "lanhuapp.com":
            return httpx.Response(302, headers={"Location": destination})
        return httpx.Response(200, content=b"safe")

    _mock_network(monkeypatch, handler)
    extractor = server.LanhuExtractor()
    try:
        if "attacker.invalid" in destination:
            with pytest.raises(ValueError, match="approved HTTPS"):
                await extractor.client.get("https://lanhuapp.com/start")
            assert len(requests) == 1
        else:
            await extractor.client.get("https://lanhuapp.com/start")
            assert len(requests) == 2
            assert "Cookie" not in requests[1].headers
            assert "Authorization" not in requests[1].headers
    finally:
        await extractor.close()


@pytest.mark.asyncio
async def test_html_mapping_path_is_validated_before_download(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"pages": {
            "../outside.html": {"html": {"sign_md5": "file.html"}},
        }})

    _mock_network(monkeypatch, handler)
    extractor = server.LanhuExtractor()

    async def document_info(*args, **kwargs):
        return {"versions": [{"id": "v1", "json_url": "https://axure-file.lanhuapp.com/mapping.json"}]}

    monkeypatch.setattr(extractor, "get_document_info", document_info)
    try:
        with pytest.raises(ValueError, match="cache directory"):
            await extractor.download_resources("pid=project&docId=document", str(tmp_path / "resource"))
    finally:
        await extractor.close()
    assert len(requests) == 1
    assert not (tmp_path / "outside.html").exists()


@pytest.mark.asyncio
async def test_dds_schema_download_does_not_send_credentials_to_cdn(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "dds.lanhuapp.com":
            return httpx.Response(200, json={"code": "00000", "data": {
                "data_resource_url": "https://lanhu-dds-backend.oss-cn-beijing.aliyuncs.com/schema.json",
            }})
        return httpx.Response(200, json={"layers": []})

    _mock_network(monkeypatch, handler)
    extractor = server.LanhuExtractor()
    try:
        assert await extractor._fetch_dds_schema("v1") == {"layers": []}
    finally:
        await extractor.close()
    assert requests[0].headers.get("Cookie") == "session=synthetic-dds"
    assert "Cookie" not in requests[1].headers
    assert "Authorization" not in requests[1].headers
