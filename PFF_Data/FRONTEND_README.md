# 🏈 QB Archetype Analysis Frontend

A beautiful, modern web interface for your QB Archetype Analysis system!

## 🚀 Quick Start

### Option 1: Simple Startup Script
```bash
python3 start_frontend.py
```

### Option 2: Direct API Launch
```bash
python3 enhanced_api.py
```

Then open your browser to: **http://localhost:5001/frontend**

## 🌟 Features

### 📊 **Individual QB Analysis**
- Analyze any QB by name
- Get archetype classification and confidence scores
- AI-powered insights about QB performance

### ⚖️ **QB Comparison**
- Compare two QBs side-by-side
- Toggle AI analysis on/off
- Detailed statistical comparisons

### 🎯 **Strategic Analysis**
- Get scenario-specific insights
- Red zone, two-minute drill, third down situations
- AI-generated strategic recommendations

### 🔄 **Dynamic AI Switching**
- Switch between Lightweight (fast) and Qwen3:8B (detailed) analyzers
- Real-time switching without restarting the API
- Visual indicators for current analyzer

### 📈 **System Status Dashboard**
- Real-time API status monitoring
- Model loading status
- AI service availability
- Data status indicators

## 🎨 **Modern UI Features**

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Beautiful Gradients**: Modern purple-blue gradient theme
- **Smooth Animations**: Hover effects and transitions
- **Loading States**: Visual feedback during API calls
- **Error Handling**: User-friendly error messages
- **Auto-dismissing Alerts**: Clean notification system

## 🔧 **Technical Details**

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Bootstrap 5 + Custom CSS
- **Icons**: Font Awesome 6
- **Backend**: Flask API (enhanced_api.py)
- **Real-time**: Dynamic analyzer switching via API

## 📱 **Usage Tips**

1. **Start with Lightweight**: Use the lightweight analyzer for quick results
2. **Switch to Qwen**: Use Qwen3:8B for detailed, comprehensive analysis
3. **Check System Status**: Monitor the dashboard for any issues
4. **Try Different Scenarios**: Experiment with strategic analysis scenarios

## 🔗 **API Endpoints**

The frontend uses these API endpoints:
- `GET /` - Main frontend interface
- `GET /health` - Health check
- `GET /models/status` - Model status
- `GET /ai/config` - Get AI analyzer config
- `POST /ai/config` - Switch AI analyzer
- `POST /analyze/ai/qb` - Individual QB analysis
- `POST /compare` - QB comparison
- `POST /analyze/ai/strategy` - Strategic analysis

## 🎯 **Example Usage**

1. **Analyze a QB**: Enter "Dillon Gabriel" and click "Analyze QB"
2. **Compare QBs**: Enter "Dillon Gabriel" vs "Will Howard" and compare
3. **Get Strategy**: Select "Red Zone Offense" for strategic insights
4. **Switch Analyzers**: Toggle between Lightweight and Qwen3:8B

## 🚀 **Enjoy Your New Frontend!**

No more command-line complexity - just a beautiful, intuitive web interface for all your QB analysis needs!
