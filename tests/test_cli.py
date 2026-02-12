import json

from sirenolite import cli


def test_cli_main_no_config(monkeypatch):
    captured = {}

    def fake_main(config=None):
        captured["config"] = config

    monkeypatch.setattr(cli.model, "main", fake_main)
    cli.main([])
    assert captured["config"] is None


def test_cli_main_with_config(tmp_path, monkeypatch):
    overrides = {
        "simulation": {"nhrs": 12},
        "solver": {"remote": False},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(overrides))

    captured = {}

    def fake_main(config=None):
        captured["config"] = config

    monkeypatch.setattr(cli.model, "main", fake_main)
    cli.main(["--config", str(config_path)])

    config = captured["config"]
    assert config["simulation"]["nhrs"] == 12
    assert config["solver"]["remote"] is False
    assert config["simulation"]["peak_load"] == 1000.0


def test_cli_dump_config(tmp_path, monkeypatch):
    output_path = tmp_path / "sirenolite_config.json"
    captured = {"called": False}

    def fake_main(config=None):
        captured["called"] = True

    monkeypatch.setattr(cli.model, "main", fake_main)
    cli.main(["--dump-config", str(output_path)])

    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert "simulation" in data
    assert data["simulation"]["nhrs"] == 400
    assert captured["called"] is False
