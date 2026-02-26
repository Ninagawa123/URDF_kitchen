# URDF_kitchen Container Export Service

This document describes the headless HTTP wrapper around `scripts/headless_batch_export.py`.

## Service endpoints

- `GET /health`
- `POST /export`

### `POST /export` request

```json
{
  "inputPath": "/workspace/Robotarium/public/Zoo/uploaded-runs/run-123/source.urdf",
  "outputRoot": "/workspace/Robots/robots/acme/rhino/v1",
  "robotName": "rhino"
}
```

### Success response

```json
{
  "ok": true,
  "inputPath": "...",
  "outputRoot": "...",
  "robotName": "...",
  "stdoutTail": "...",
  "stderrTail": "..."
}
```

## Build + run (when Docker is available)

From `URDF_kitchen/`:

```bash
docker compose -f docker/docker-compose.export-service.yml up --build -d
```

The compose file maps:

- Host: `/home/stuart/KinematicTrees`
- Container: `/workspace`

So caller path rewrite should usually be:

- `URDF_KITCHEN_SERVICE_PATH_FROM=/home/stuart/KinematicTrees`
- `URDF_KITCHEN_SERVICE_PATH_TO=/workspace`

## Local host run (without Docker)

```bash
python3 scripts/export_service.py
```

Default bind: `0.0.0.0:8091`.
