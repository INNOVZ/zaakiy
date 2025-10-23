#!/bin/bash
# ================================
# Redis Setup Script
# ================================
# This script sets up and configures Redis for the ZaaKy platform

set -e  # Exit on error

echo "🚀 Redis Setup for ZaaKy Platform"
echo "=================================="
echo ""

# Detect OS
OS="$(uname -s)"
echo "📍 Detected OS: $OS"
echo ""

# Function to check if Redis is installed
check_redis_installed() {
    if command -v redis-server &> /dev/null; then
        echo "✅ Redis server is installed"
        redis-server --version
        return 0
    else
        echo "❌ Redis server is NOT installed"
        return 1
    fi
}

# Function to install Redis on macOS
install_redis_macos() {
    echo "📦 Installing Redis on macOS..."
    if command -v brew &> /dev/null; then
        brew install redis
        echo "✅ Redis installed successfully via Homebrew"
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
}

# Function to install Redis on Linux
install_redis_linux() {
    echo "📦 Installing Redis on Linux..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y redis-server
        echo "✅ Redis installed successfully via apt-get"
    elif command -v yum &> /dev/null; then
        sudo yum install -y redis
        echo "✅ Redis installed successfully via yum"
    else
        echo "❌ Package manager not found. Please install Redis manually."
        exit 1
    fi
}

# Check if Redis is installed
if ! check_redis_installed; then
    echo ""
    read -p "Do you want to install Redis now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        case "$OS" in
            Darwin*)
                install_redis_macos
                ;;
            Linux*)
                install_redis_linux
                ;;
            *)
                echo "❌ Unsupported OS: $OS"
                echo "Please install Redis manually: https://redis.io/download"
                exit 1
                ;;
        esac
    else
        echo "❌ Redis installation cancelled"
        exit 1
    fi
fi

echo ""
echo "🔧 Configuring Redis..."

# Check if Redis is running
if redis-cli ping &> /dev/null; then
    echo "✅ Redis is already running"
else
    echo "⚠️  Redis is not running. Starting Redis..."

    case "$OS" in
        Darwin*)
            # macOS - start with Homebrew services
            if command -v brew &> /dev/null; then
                brew services start redis
                echo "✅ Redis started via Homebrew services"
            else
                redis-server --daemonize yes
                echo "✅ Redis started in daemon mode"
            fi
            ;;
        Linux*)
            # Linux - start with systemd
            if command -v systemctl &> /dev/null; then
                sudo systemctl start redis
                sudo systemctl enable redis
                echo "✅ Redis started and enabled via systemd"
            else
                redis-server --daemonize yes
                echo "✅ Redis started in daemon mode"
            fi
            ;;
    esac

    # Wait for Redis to start
    sleep 2
fi

# Test Redis connection
echo ""
echo "🧪 Testing Redis connection..."
if redis-cli ping > /dev/null 2>&1; then
    REDIS_VERSION=$(redis-cli --version | awk '{print $2}')
    echo "✅ Redis connection successful!"
    echo "   Version: $REDIS_VERSION"

    # Test basic operations
    echo ""
    echo "🧪 Testing Redis operations..."
    redis-cli SET test_key "ZaaKy Platform" > /dev/null
    VALUE=$(redis-cli GET test_key)
    redis-cli DEL test_key > /dev/null

    if [ "$VALUE" = "ZaaKy Platform" ]; then
        echo "✅ Redis read/write test successful"
    else
        echo "⚠️  Redis read/write test failed"
    fi

    # Display Redis info
    echo ""
    echo "📊 Redis Server Info:"
    redis-cli INFO server | grep -E "redis_version|os|uptime_in_seconds|tcp_port"

    echo ""
    echo "📊 Redis Memory Usage:"
    redis-cli INFO memory | grep -E "used_memory_human|maxmemory"

else
    echo "❌ Redis connection failed"
    echo "   Please check if Redis is running: redis-cli ping"
    exit 1
fi

# Configure eviction policy
echo ""
echo "🔧 Configuring Redis eviction policy..."
redis-cli CONFIG SET maxmemory-policy allkeys-lru > /dev/null
echo "✅ Set eviction policy to allkeys-lru"

# Set reasonable memory limit (optional)
redis-cli CONFIG SET maxmemory 256mb > /dev/null
echo "✅ Set max memory to 256MB (adjust as needed)"

echo ""
echo "================================================"
echo "✅ Redis Setup Complete!"
echo "================================================"
echo ""
echo "📝 Next Steps:"
echo "   1. Update your .env file with Redis configuration:"
echo "      REDIS_URL=redis://localhost:6379"
echo "      REDIS_PASSWORD="
echo ""
echo "   2. Restart your application to use Redis caching"
echo ""
echo "🔍 Useful Commands:"
echo "   - Check Redis status:  redis-cli ping"
echo "   - Monitor Redis:       redis-cli monitor"
echo "   - Redis CLI:           redis-cli"
echo "   - View stats:          redis-cli INFO"
echo "   - Stop Redis (macOS):  brew services stop redis"
echo "   - Stop Redis (Linux):  sudo systemctl stop redis"
echo ""
