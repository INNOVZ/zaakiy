# 🚀 ZaaKy AI Platform - Production Deployment Guide

This guide covers multiple deployment options for the ZaaKy AI Platform backend.

## 📋 Prerequisites

Before deploying, ensure you have:

- **Required API Keys:**

  - Supabase URL, Service Role Key, JWT Secret
  - OpenAI API Key
  - Pinecone API Key and Index Name

- **System Requirements:**
  - Docker and Docker Compose
  - At least 2GB RAM
  - 10GB disk space
  - Python 3.11+ (for local development)

## 🐳 Option 1: Docker Deployment (Recommended)

### Quick Start

1. **Configure Environment:**

   ```bash
   cp .env.production .env
   # Edit .env with your production values
   ```

2. **Deploy:**

   ```bash
   ./deploy.sh
   ```

3. **Verify Deployment:**
   ```bash
   curl http://localhost:8001/health
   ```

### Manual Docker Deployment

1. **Build and Run:**

   ```bash
   # Build image
   docker build -t zaaky-backend .

   # Run with production compose
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Check Status:**
   ```bash
   docker-compose -f docker-compose.prod.yml ps
   docker-compose -f docker-compose.prod.yml logs -f
   ```

### Production with Nginx + SSL

1. **Generate SSL Certificates:**

   ```bash
   mkdir ssl
   # Add your SSL certificates to ssl/ directory
   # cert.pem and key.pem
   ```

2. **Deploy with SSL:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## ☁️ Option 2: Cloud Platform Deployment

### Railway.app

1. **Connect Repository:**

   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will automatically detect the `railway.json` configuration

2. **Set Environment Variables:**
   - Add all required environment variables in Railway dashboard
   - Deploy automatically triggers

### Vercel

1. **Install Vercel CLI:**

   ```bash
   npm i -g vercel
   ```

2. **Deploy:**

   ```bash
   vercel --prod
   ```

3. **Set Environment Variables:**
   - Add variables in Vercel dashboard
   - Redeploy after adding variables

### DigitalOcean App Platform

1. **Create App:**

   - Go to DigitalOcean App Platform
   - Connect your repository
   - Select "Docker" as source type

2. **Configure:**
   - Set build command: `docker build -t zaaky-backend .`
   - Set run command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

### AWS ECS/Fargate

1. **Create ECR Repository:**

   ```bash
   aws ecr create-repository --repository-name zaaky-backend
   ```

2. **Build and Push:**

   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

   # Build and tag
   docker build -t zaaky-backend .
   docker tag zaaky-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/zaaky-backend:latest

   # Push
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/zaaky-backend:latest
   ```

3. **Create ECS Task Definition:**
   - Use the pushed image
   - Configure environment variables
   - Set memory and CPU limits

### Google Cloud Run

1. **Build and Push:**

   ```bash
   # Build
   docker build -t gcr.io/PROJECT-ID/zaaky-backend .

   # Push
   docker push gcr.io/PROJECT-ID/zaaky-backend
   ```

2. **Deploy:**
   ```bash
   gcloud run deploy zaaky-backend \
     --image gcr.io/PROJECT-ID/zaaky-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## 🔧 Option 3: Traditional Server Deployment

### Ubuntu/Debian Server

1. **Install Dependencies:**

   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python 3.11
   sudo apt install python3.11 python3.11-venv python3.11-dev -y

   # Install system dependencies
   sudo apt install build-essential curl nginx -y
   ```

2. **Setup Application:**

   ```bash
   # Clone repository
   git clone <your-repo-url> /opt/zaaky-backend
   cd /opt/zaaky-backend

   # Create virtual environment
   python3.11 -m venv .venv
   source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment:**

   ```bash
   cp .env.production .env
   # Edit .env with your values
   ```

4. **Setup Systemd Service:**

   ```bash
   sudo tee /etc/systemd/system/zaaky-backend.service > /dev/null <<EOF
   [Unit]
   Description=ZaaKy AI Platform Backend
   After=network.target

   [Service]
   Type=exec
   User=www-data
   Group=www-data
   WorkingDirectory=/opt/zaaky-backend
   Environment=PATH=/opt/zaaky-backend/.venv/bin
   ExecStart=/opt/zaaky-backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   EOF

   # Enable and start service
   sudo systemctl daemon-reload
   sudo systemctl enable zaaky-backend
   sudo systemctl start zaaky-backend
   ```

5. **Configure Nginx:**

   ```bash
   # Copy nginx configuration
   sudo cp nginx.prod.conf /etc/nginx/sites-available/zaaky-backend
   sudo ln -s /etc/nginx/sites-available/zaaky-backend /etc/nginx/sites-enabled/
   sudo rm /etc/nginx/sites-enabled/default

   # Test and reload nginx
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## 🔒 Security Considerations

### Environment Variables

- **Never commit `.env` files to version control**
- Use environment-specific files (`.env.production`, `.env.staging`)
- Rotate API keys regularly
- Use secrets management services in production

### SSL/TLS

- Always use HTTPS in production
- Configure proper SSL certificates
- Enable HSTS headers
- Use strong cipher suites

### Firewall

```bash
# Basic UFW configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Database Security

- Use connection pooling
- Enable SSL for database connections
- Implement proper backup strategies
- Monitor database performance

## 📊 Monitoring and Maintenance

### Health Checks

- **Basic Health:** `GET /health`
- **Detailed Health:** `GET /health/detailed`
- **Client Health:** `GET /health/clients`

### Logging

- Application logs: `./logs/`
- Nginx logs: `/var/log/nginx/`
- System logs: `journalctl -u zaaky-backend`

### Performance Monitoring

```bash
# Check container resource usage
docker stats

# Check application performance
curl http://localhost:8001/health/detailed

# Monitor logs
tail -f logs/zaaky_$(date +%Y%m%d).log
```

### Backup Strategy

1. **Database Backups:**

   - Configure Supabase automated backups
   - Export data regularly

2. **Application Backups:**
   - Backup configuration files
   - Backup uploaded documents
   - Backup logs for analysis

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use:**

   ```bash
   sudo lsof -i :8001
   sudo kill -9 <PID>
   ```

2. **Permission Denied:**

   ```bash
   sudo chown -R www-data:www-data /opt/zaaky-backend
   sudo chmod +x deploy.sh
   ```

3. **Environment Variables Not Loading:**

   - Check `.env` file exists and has correct format
   - Verify no spaces around `=` in environment variables
   - Restart application after changes

4. **Database Connection Issues:**
   - Verify Supabase credentials
   - Check network connectivity
   - Review firewall settings

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Performance Issues

1. **High Memory Usage:**

   - Reduce worker count
   - Increase memory limits
   - Check for memory leaks

2. **Slow Response Times:**
   - Enable compression
   - Optimize database queries
   - Check external API response times

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer:**

   - Use nginx or cloud load balancer
   - Configure sticky sessions if needed
   - Health check all instances

2. **Multiple Instances:**
   ```bash
   # Scale with docker-compose
   docker-compose -f docker-compose.prod.yml up -d --scale zaaky-backend=3
   ```

### Vertical Scaling

1. **Increase Resources:**

   - More CPU cores
   - More RAM
   - Faster storage (SSD)

2. **Optimize Configuration:**
   - Tune worker processes
   - Optimize database connections
   - Enable caching

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Build new image
docker build -t zaaky-backend:latest .

# Update containers one by one
docker-compose -f docker-compose.prod.yml up -d --no-deps zaaky-backend
```

### Database Migrations

```bash
# Run migrations
python scripts/migrate_data.py --execute

# Backup before migration
python scripts/backup_data.py
```

### Monitoring Updates

- Set up alerts for health check failures
- Monitor resource usage trends
- Track error rates and response times
- Regular security updates

---

## 🆘 Support

If you encounter issues during deployment:

1. Check the logs: `docker-compose logs -f`
2. Verify environment variables
3. Test health endpoints
4. Review this documentation
5. Contact support: support@zaaky.ai

**Happy Deploying! 🚀**
