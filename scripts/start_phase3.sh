#!/bin/bash
# Start Phase 3 development environment

echo "🚀 Starting LendenClub Voice Assistant - Phase 3"
echo "=============================================="

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Start backend in background
echo "🔧 Starting backend services..."
python main_text_pipeline.py --api-mode &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🌐 Starting frontend development server..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "✅ Services started!"
echo "📱 Frontend: http://localhost:3000"
echo "🔗 Backend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interruption
trap 'kill $BACKEND_PID $FRONTEND_PID' INT
wait