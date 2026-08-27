def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "vllm_qwen38_27b" in r.json()
