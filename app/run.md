From the backend/ directory:


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
(--port 8000 matches NEXT_PUBLIC_API_BASE_URL in frontend/.env.local.) First-time setup if you haven't already:


pip install -r requirements.txt
Note this installs the full local frontrunner stack (torch, ultralytics, faster-whisper, faiss) which is heavy — and you'll need ffmpeg on PATH for audio extraction, since it isn't installed in this sandbox yet. Once it's up, GET http://localhost:8000/health should show frontrunner_ready: true and qwen_ready: true.

---

## Deploying to an Azure VM (Docker + nginx + certbot)

Target: `swarm.recruitbase.work` -> Azure VM, container bound to localhost only,
nginx on the host terminates TLS and reverse-proxies to it. Run everything
below from your own machine unless a step says "on the VM".

### 0. Prerequisites

- An Azure VM (Ubuntu 22.04/24.04 recommended) with a public IP, SSH access,
  and a Network Security Group (NSG) you can edit.
- This repo pushed to GitHub — `.github/workflows/backend-docker-publish.yml`
  builds `backend/Dockerfile` and publishes it to GHCR on every push to
  `main` that touches `backend/**` (also runs as a build-only check on PRs).
  No extra setup needed on the GitHub side; it authenticates with the
  workflow's own `GITHUB_TOKEN`.
- DNS access for `recruitbase.work` to add an A record.

GHCR packages publish **private** by default. Either flip the package to
public (github.com -> your profile/org -> Packages -> the package -> Package
settings -> Change visibility), or plan on `docker login ghcr.io` with a PAT
on the VM in step 5.

### 1. Open ports 80/443 on the VM's NSG

Azure Portal: VM -> Networking -> Add inbound port rule -> allow TCP 80 and
443 from `Any` (source). Or via CLI from your machine:

```bash
az vm open-port --resource-group <RESOURCE_GROUP> --name <VM_NAME> --port 80 --priority 900
az vm open-port --resource-group <RESOURCE_GROUP> --name <VM_NAME> --port 443 --priority 901
```

Port 22 (SSH) should already be open from VM creation — don't widen it beyond
what you need.

### 2. Point the domain at the VM

At your DNS provider for `recruitbase.work`, add:

```
A    swarm    <VM_PUBLIC_IP>    TTL 300
```

Wait for it to propagate (`dig swarm.recruitbase.work` should return the VM's
IP) before running certbot in step 7 — it validates ownership over HTTP.

### 3. Build and push the image

Push to `main` (or merge a PR into it) and let CI do this:

```bash
git push origin main
```

Watch it under the repo's **Actions** tab. Once it's green, the image is at
`ghcr.io/<your-github-username>/<repo-name>:latest`.

Don't want to wait on CI, or testing before pushing? Build and push it
yourself the same way the workflow does:

```bash
cd backend
docker login ghcr.io -u <your-github-username>   # PAT with write:packages, as the password
docker build -t ghcr.io/<your-github-username>/<repo-name>:latest .
docker push ghcr.io/<your-github-username>/<repo-name>:latest
```

### 4. SSH into the VM and install Docker

```bash
ssh -i <PATH_TO_KEY>.pem <SSH_USER>@<VM_PUBLIC_IP>

# on the VM:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker                      # or log out/in for the group change to apply
```

### 5. Pull the image and create the persistent cache volume (on the VM)

```bash
docker login ghcr.io -u <your-github-username>   # PAT with read:packages, as the password — skip if the package is public
docker pull ghcr.io/<your-github-username>/<repo-name>:latest
docker volume create swarm-audiences-cache
```

The named volume is mounted at `/home/appuser/.cache` inside the container —
that's where YOLO/Whisper/CLAP weights land on first run, so they persist
across container restarts/upgrades instead of re-downloading every time
(matches the same ownership-fix entrypoint used in the Dockerfile).

### 6. Create the env file and run the container (on the VM)

Don't commit secrets — create this file directly on the VM (`nano` or `scp`
it over from your local `backend/.env.local`, never via git):

```bash
sudo mkdir -p /opt/swarm-audiences
sudo nano /opt/swarm-audiences/.env
```

```env
QWEN_MODEL_CLOUD_API_URL=https://ws-kz1wc90xu7l1tmnd.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_CLOUD_API_KEY=sk-ws-...
QWEN_MODEL_NAME=qwen3-vl-flash
CORS_ORIGINS=https://<your-frontend-domain>
```

Then run the container bound to `127.0.0.1` only — nginx is the one thing
that should be reachable from the public internet, terminating TLS before
traffic ever reaches uvicorn:

```bash
docker run -d \
  --name swarm-audiences-backend \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v swarm-audiences-cache:/home/appuser/.cache \
  --env-file /opt/swarm-audiences/.env \
  ghcr.io/<your-github-username>/<repo-name>:latest

docker logs -f swarm-audiences-backend   # watch startup; first run downloads model weights
```

If the VM has an NVIDIA GPU (an Azure NC-series SKU), additionally install
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
add `--gpus all` to the `docker run` command, and set `FRONTRUNNER_DEVICE=cuda`
in the `.env` file above. Without a GPU, the code already falls back to CPU
automatically — no extra flags needed.

### 7. Install nginx + certbot and issue the certificate (on the VM)

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

```bash
sudo tee /etc/nginx/sites-available/swarm.recruitbase.work > /dev/null <<'EOF'
server {
    listen 80;
    server_name swarm.recruitbase.work;

    # video uploads can be large; match backend's MAX_UPLOAD_BYTES (500MB default)
    client_max_body_size 500m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # deep-audit calls can run long (Qwen round trip); don't cut them off
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/swarm.recruitbase.work /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

```bash
sudo certbot --nginx -d swarm.recruitbase.work -m <YOUR_EMAIL> --agree-tos --redirect
```

Certbot rewrites the nginx config in place to add the 443/TLS server block
and an HTTP -> HTTPS redirect, and installs a systemd timer for renewal.
Verify renewal works without actually renewing:

```bash
sudo certbot renew --dry-run
systemctl status certbot.timer
```

### 8. Verify

```bash
curl https://swarm.recruitbase.work/health
```

Should return `{"status":"ok","frontrunner_ready":true,"qwen_ready":true,...}`
once the first-run model downloads finish (watch `docker logs -f
swarm-audiences-backend` on the VM if it's slow).

### 9. Redeploying after a code change

```bash
# from your dev machine — CI rebuilds and republishes :latest
git push origin main

# on the VM, once the Actions run is green
docker pull ghcr.io/<your-github-username>/<repo-name>:latest
docker stop swarm-audiences-backend && docker rm swarm-audiences-backend
docker run -d \
  --name swarm-audiences-backend \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v swarm-audiences-cache:/home/appuser/.cache \
  --env-file /opt/swarm-audiences/.env \
  ghcr.io/<your-github-username>/<repo-name>:latest
```

The named volume survives `stop`/`rm`, so model weights aren't re-downloaded
on redeploy.