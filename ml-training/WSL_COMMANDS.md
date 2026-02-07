# WSL Docker Commands Reference

## unsloth

Starts Jupyter Lab in the Unsloth Docker container with GPU support.

```bash
unsloth          # Run with cached image (fast)
unsloth --new    # Rebuild image from Dockerfile first, then run
```

**What it does:**
- Mounts `ml-training/` folder to `/workspace/` in container
- Enables GPU access (`--gpus all`)
- Runs Jupyter Lab on port 8888
- Access at: `http://localhost:8888`

**Use `--new` when:**
- You modified the Dockerfile
- You want to add new packages to the container

---

## Dockerfile Location

```
ml-training/Dockerfile
```

To add packages, edit the Dockerfile:
```dockerfile
FROM unsloth/unsloth:latest
USER root
RUN apt-get update && apt-get install -y wget YOUR_PACKAGE_HERE && rm -rf /var/lib/apt/lists/*
```

Then run `unsloth --new` to rebuild.

---

## Other Commands

```bash
wsl-shutdown     # Completely shut down WSL (defined in ~/.bashrc)
docker ps        # See running containers
docker stop ID   # Stop a container
```

---

## Path Translations

| Context | Path Style | Example |
|---------|------------|---------|
| Windows/Claude Code | `C:\Users\david\...` | `C:\Users\david\RLM_TEST\ml-training` |
| WSL | `/mnt/c/Users/david/...` | `/mnt/c/Users/david/RLM_TEST/ml-training` |
| Inside Docker | `/workspace/...` | `/workspace/data/file.json` |
