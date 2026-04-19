#!/bin/bash
# =============================================================================
# Danfoss RAG Chatbot - VPS Deployment Script
# Run this on your Hostinger VPS
# =============================================================================
set -e

DOMAIN="rag.srv844085.hstgr.cloud"
APP_DIR="/opt/danfoss-rag"
NGINX_CONF="/etc/nginx/sites-available/$DOMAIN"

echo "============================================"
echo "  Danfoss RAG Chatbot - Deployment"
echo "============================================"

# ---- Step 1: Check prerequisites ----
echo ""
echo "[1/6] Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Install it first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: docker-compose is not installed. Install it first."
    exit 1
fi

if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    apt-get update && apt-get install -y nginx
fi

echo "  ✓ Docker, docker-compose, and Nginx are available"

# ---- Step 2: Set up application directory ----
echo ""
echo "[2/6] Setting up application directory..."

if [ ! -d "$APP_DIR" ]; then
    echo "  Creating $APP_DIR..."
    mkdir -p "$APP_DIR"
    echo "  IMPORTANT: Copy your project files to $APP_DIR"
    echo "  You can use scp, git clone, or Hostinger file manager"
    echo ""
    echo "  Example with scp (run from your Windows machine):"
    echo "    scp -r ./* root@YOUR_VPS_IP:$APP_DIR/"
    echo ""
    echo "  Or if you have git:"
    echo "    cd $APP_DIR && git clone YOUR_REPO_URL ."
fi

# Check if project files exist
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    echo ""
    echo "  WARNING: docker-compose.yml not found in $APP_DIR"
    echo "  Please copy your project files there first, then re-run this script."
    exit 1
fi

echo "  ✓ Project files found"

# ---- Step 3: Check .env file ----
echo ""
echo "[3/6] Checking environment configuration..."

if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/.env.production" ]; then
        cp "$APP_DIR/.env.production" "$APP_DIR/.env"
        echo "  Copied .env.production → .env"
        echo ""
        echo "  !! IMPORTANT: Edit $APP_DIR/.env and add your real API keys !!"
        echo "  Run: nano $APP_DIR/.env"
        echo ""
        read -p "  Have you filled in the API keys? (y/n): " answer
        if [ "$answer" != "y" ]; then
            echo "  Please edit .env first, then re-run this script."
            exit 1
        fi
    else
        echo "  ERROR: No .env or .env.production found."
        echo "  Copy .env.production to $APP_DIR/.env and fill in your API keys."
        exit 1
    fi
fi

echo "  ✓ .env file exists"

# ---- Step 4: Create uploads directory ----
mkdir -p "$APP_DIR/uploads"

# ---- Step 5: Build and start Docker containers ----
echo ""
echo "[4/6] Building and starting Docker containers..."

cd "$APP_DIR"

# Use 'docker compose' (v2) or 'docker-compose' (v1)
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD down 2>/dev/null || true
$COMPOSE_CMD up -d --build

echo "  ✓ Container started"

# Wait for health check
echo "  Waiting for application to become healthy..."
for i in {1..30}; do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "  ✓ Application is healthy!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  WARNING: Health check timed out. Check logs:"
        echo "    $COMPOSE_CMD logs backend"
    fi
    sleep 2
done

# ---- Step 6: Configure Nginx ----
echo ""
echo "[5/6] Configuring Nginx reverse proxy..."

# Create sites-available/enabled dirs if they don't exist
mkdir -p /etc/nginx/sites-available
mkdir -p /etc/nginx/sites-enabled

# Check if nginx uses sites-enabled (some setups use conf.d)
if ! grep -q "sites-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
    echo "  Adding sites-enabled include to nginx.conf..."
    # Add include directive before the last closing brace
    if ! grep -q "include /etc/nginx/sites-enabled" /etc/nginx/nginx.conf; then
        sed -i '/^http {/a\    include /etc/nginx/sites-enabled/*;' /etc/nginx/nginx.conf 2>/dev/null || true
    fi
fi

cp "$APP_DIR/deploy/nginx-rag.conf" "$NGINX_CONF"

# Create symlink if not exists
if [ ! -L "/etc/nginx/sites-enabled/$DOMAIN" ]; then
    ln -sf "$NGINX_CONF" "/etc/nginx/sites-enabled/$DOMAIN"
fi

# Test and reload Nginx
if nginx -t 2>/dev/null; then
    systemctl reload nginx
    echo "  ✓ Nginx configured and reloaded"
else
    echo "  ERROR: Nginx config test failed. Check: nginx -t"
    exit 1
fi

# ---- Step 7: SSL with Certbot ----
echo ""
echo "[6/6] SSL Certificate..."

if command -v certbot &> /dev/null; then
    echo "  Certbot found. Setting up SSL..."
    echo "  Make sure DNS for $DOMAIN points to this server first!"
    echo ""
    read -p "  Run certbot now? (y/n): " ssl_answer
    if [ "$ssl_answer" = "y" ]; then
        certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@example.com || {
            echo "  Certbot failed. You can run it manually later:"
            echo "    certbot --nginx -d $DOMAIN"
        }
    fi
else
    echo "  Certbot not found. Install it for SSL:"
    echo "    apt-get install certbot python3-certbot-nginx"
    echo "    certbot --nginx -d $DOMAIN"
fi

# ---- Done ----
echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  App URL:     http://$DOMAIN"
echo "  Health:      http://$DOMAIN/health"
echo "  API Docs:    http://$DOMAIN/docs"
echo "  Chat Widget: http://$DOMAIN/static/demo.html"
echo ""
echo "  Useful commands:"
echo "    cd $APP_DIR"
echo "    $COMPOSE_CMD logs -f        # View logs"
echo "    $COMPOSE_CMD restart        # Restart"
echo "    $COMPOSE_CMD down           # Stop"
echo "    $COMPOSE_CMD up -d --build  # Rebuild"
echo ""
echo "  Next steps:"
echo "    1. Point DNS: A record for $DOMAIN → this server's IP"
echo "    2. Run certbot for SSL (if not done above)"
echo "    3. Ingest documents via /api/ingest or locally"
echo "============================================"
