# AWS EC2 Deployment Guide

This guide provides step-by-step instructions for deploying the Anime Recommender RAG Model API to AWS EC2.

## Prerequisites

- AWS Account
- AWS CLI configured (optional, but helpful)
- SSH key pair for EC2 access
- Basic knowledge of Linux commands

## Step 1: Launch EC2 Instance

### 1.1 Create EC2 Instance

1. Log in to AWS Console
2. Navigate to **EC2** service
3. Click **Launch Instance**
4. Configure the instance:

   - **Name**: `anime-recommender-api` (or your preferred name)
   - **AMI**: Ubuntu 22.04 LTS (recommended)
   - **Instance Type**:
     - For testing: `t3.medium` or `t3.large` (2-4 vCPU, 4-8 GB RAM)
     - For production: `t3.xlarge` or larger (more memory for embeddings)
   - **Key Pair**: Select or create a new key pair (download the `.pem` file)
   - **Network Settings**:
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere (0.0.0.0/0)
     - Allow Custom TCP (port 8000) from anywhere (0.0.0.0/0) for API access
   - **Storage**:
     - Minimum 20 GB (recommend 30-50 GB for vector store and models)
   - **Security Group**: Create new or use existing
     - **Inbound Rules**:
       - SSH (22) from your IP
       - HTTP (80) from anywhere
       - Custom TCP (8000) from anywhere

5. Click **Launch Instance**

### 1.2 Note Instance Details

- **Public IP**: You'll need this to SSH into the instance
- **Instance ID**: For reference

## Step 2: Connect to EC2 Instance

### 2.1 SSH into Instance

```bash
# On Windows (PowerShell) or Linux/Mac
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_IP>

# Example:
ssh -i ~/.ssh/anime-recommender.pem ubuntu@54.123.45.67
```

**Note**: If you're on Windows, you might need to use WSL or Git Bash, or convert the key using PuTTY.

### 2.2 Update System

```bash
sudo apt update
sudo apt upgrade -y
```

## Step 3: Install Docker

### 3.1 Install Docker

```bash
# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index
sudo apt update

# Install Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
# Or run: newgrp docker
```

### 3.2 Verify Docker Installation

```bash
docker --version
docker run hello-world
```

## Step 4: Install Docker Compose (Optional but Recommended)

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make it executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

## Step 5: Transfer Project Files to EC2

### 5.1 Option A: Using SCP (from your local machine)

```bash
# Create a tarball of the project (excluding chroma_db and venv)
tar --exclude='chroma_db' --exclude='venv' --exclude='.git' -czf rag_model.tar.gz rag_model/

# Transfer to EC2
scp -i /path/to/your-key.pem rag_model.tar.gz ubuntu@<PUBLIC_IP>:~/

# SSH into EC2 and extract
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_IP>
tar -xzf rag_model.tar.gz
cd rag_model
```

### 5.2 Option B: Using Git (if project is in a repository)

```bash
# On EC2 instance
sudo apt install -y git
git clone <your-repo-url>
cd rag_model
```

### 5.3 Option C: Using AWS CodeDeploy or S3

Upload project files to S3 and download on EC2, or use AWS CodeDeploy for automated deployments.

## Step 6: Setup Environment Variables

### 6.1 Create .env File

```bash
cd rag_model
cp .env.example .env
nano .env  # or use vi/vim
```

### 6.2 Configure .env File

Edit the `.env` file with your actual values:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
HUGGINGFACE_API_TOKEN=
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=Anime_embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0
CSV_FILE_PATH=./data/Anime_Cleaned.csv
CHUNK_SIZE=1000
CHUNK_OVERLAP=0
BATCH_SIZE=100
SEARCH_K=1000
```

**Important**: Make sure `GROQ_API_KEY` is set correctly.

## Step 7: Initialize Vector Store

### 7.1 Build Docker Image

```bash
cd rag_model
docker build -t anime-recommender .
```

### 7.2 Initialize Database

```bash
# Run initialization script in Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender \
  python scripts/initialize_db.py
```

**Note**: This will take 15-20 minutes. The process will:

- Load the CSV file
- Generate embeddings
- Store them in ChromaDB

## Step 8: Run the Application

### 8.1 Run Docker Container

```bash
docker run -d \
  --name anime-recommender \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender
```

### 8.2 Verify Container is Running

```bash
docker ps
docker logs anime-recommender
```

### 8.3 Test the API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test recommendation endpoint
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Naruto"}'
```

## Step 9: Configure Public Access

### 9.1 Test from Public IP

From your local machine, test using the EC2 public IP:

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
```

### 9.2 Verify Security Group

Ensure your EC2 security group allows:

- **Inbound Rule**: Custom TCP port 8000 from 0.0.0.0/0 (anywhere)

### 9.3 Test Recommendation API

```bash
curl -X POST "http://<EC2_PUBLIC_IP>:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "action anime"}'
```

## Step 10: Setup Nginx Reverse Proxy (Optional but Recommended)

For production, it's recommended to use Nginx as a reverse proxy.

### 10.1 Install Nginx

```bash
sudo apt install -y nginx
```

### 10.2 Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/anime-recommender
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name <YOUR_DOMAIN_OR_IP>;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 10.3 Enable Site

```bash
sudo ln -s /etc/nginx/sites-available/anime-recommender /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 10.4 Update Security Group

- Remove port 8000 from public access
- Keep port 80 (HTTP) open
- Optionally add port 443 (HTTPS) for SSL

## Step 11: Setup SSL with Let's Encrypt (Optional)

### 11.1 Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 11.2 Obtain SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com
```

Follow the prompts to complete the setup.

## Step 12: Monitoring and Maintenance

### 12.1 View Logs

```bash
# Docker logs
docker logs anime-recommender
docker logs -f anime-recommender  # Follow logs

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 12.2 Restart Container

```bash
docker restart anime-recommender
```

### 12.3 Update Application

```bash
# Pull latest code
git pull  # if using git

# Rebuild and restart
docker stop anime-recommender
docker rm anime-recommender
docker build -t anime-recommender .
docker run -d \
  --name anime-recommender \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs anime-recommender

# Check if port is in use
sudo netstat -tulpn | grep 8000

# Check environment variables
docker run --rm --env-file .env anime-recommender env | grep GROQ
```

### Vector Store Not Found

```bash
# Reinitialize database
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender \
  python scripts/initialize_db.py
```

### Out of Memory

If you encounter memory issues:

- Use a larger instance type (t3.xlarge or larger)
- Reduce `BATCH_SIZE` in `.env`
- Monitor memory usage: `free -h`

### API Not Accessible

- Check security group rules
- Check EC2 instance firewall: `sudo ufw status`
- Verify container is running: `docker ps`
- Test locally on EC2: `curl http://localhost:8000/health`

## Cost Optimization

- Use **Spot Instances** for development/testing
- Stop instance when not in use
- Use **Reserved Instances** for production (1-year or 3-year commitment)
- Monitor CloudWatch for usage and costs

## Security Best Practices

1. **Never commit `.env` file** to version control
2. **Use IAM roles** instead of hardcoding credentials
3. **Enable CloudWatch logging** for monitoring
4. **Regularly update** system packages and Docker images
5. **Use HTTPS** in production (Let's Encrypt SSL)
6. **Restrict SSH access** to your IP only
7. **Use AWS Secrets Manager** for API keys in production

## Next Steps

- Set up automated deployments (CI/CD)
- Configure CloudWatch alarms
- Set up auto-scaling if needed
- Implement rate limiting
- Add authentication if required

## Support

For issues or questions:

- Check application logs: `docker logs anime-recommender`
- Check system resources: `htop` or `free -h`
- Review AWS CloudWatch logs
- Verify environment variables are set correctly
