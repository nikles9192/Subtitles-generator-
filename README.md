# 🎧 Audio to SRT Caption Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Transform your audio files into perfectly timed SRT subtitles with AI-powered transcription!**

A modern, user-friendly web application that converts audio files to synchronized subtitle files using OpenAI's Whisper models. Built with Streamlit for an intuitive interface and powered by Hugging Face Transformers.

## ✨ Features

🎯 **Smart Transcription**
- Multiple Whisper model options (tiny to large-v3)
- English-optimized models for better performance
- Word-level timestamp accuracy
- Intelligent sentence boundary detection

📝 **Advanced Caption Formatting**
- Customizable character limits per caption line
- Smart text splitting at punctuation marks
- Professional SRT format output
- Readable subtitle timing

🖥️ **User-Friendly Interface**
- Drag & drop audio file upload
- Real-time audio preview
- Live caption preview
- One-click SRT download

🔧 **Technical Excellence**
- GPU acceleration support
- Efficient model caching
- Multiple audio format support (WAV, MP3, M4A, OGG)
- Responsive web design

## 🎬 Supported Models

### 🌍 Multilingual Models
- `whisper-tiny` - Fastest, basic accuracy
- `whisper-base` - Balanced speed and accuracy ⭐ **Recommended**
- `whisper-small` - Good accuracy, moderate speed
- `whisper-medium` - High accuracy, slower processing
- `whisper-large-v2` - Excellent accuracy, slow
- `whisper-large-v3` - Best accuracy, slowest

### 🇺🇸 English-Only Models (Optimized)
- `whisper-tiny.en` - Fastest English transcription
- `whisper-base.en` - Balanced English performance
- `whisper-small.en` - High-quality English results
- `whisper-medium.en` - Premium English accuracy

> 💡 **Tip:** English-only models are faster and more accurate for English audio!

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/subtitles-generator.git
   cd subtitles-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   The app will automatically open at `http://localhost:8501`

## 🚀 Usage Guide

### Step 1: Choose Your Model 🎯
- Select from the sidebar dropdown
- **For English audio:** Use `.en` models for best results
- **For other languages:** Use standard multilingual models
- **Recommended:** Start with `whisper-base` or `whisper-base.en`

### Step 2: Upload Audio File 📁
- Drag and drop or click to browse
- Supported formats: WAV, MP3, M4A, OGG
- Preview your audio before processing

### Step 3: Customize Settings ⚙️
- Adjust **Max Characters per Caption** (20-100)
- Default: 42 characters (optimal for readability)
- Preview how your captions will look

### Step 4: Generate Captions ✨
- Click "Generate SRT Captions"
- Wait for AI processing (time varies by model and audio length)
- Review generated captions in the preview area

### Step 5: Download & Use 📥
- Download your `.srt` file
- Import into video editors, players, or streaming platforms
- Professional-quality subtitles ready to use!

## 📋 Requirements

```
streamlit>=1.25.0
torch>=1.13.0
librosa>=0.9.2
transformers>=4.21.0
```

## 🎨 Example Output

```srt
1
00:00:00,000 --> 00:00:03,240
Welcome to our audio transcription demo.

2
00:00:03,240 --> 00:00:06,120
This tool creates perfect subtitles.

3
00:00:06,120 --> 00:00:09,480
Try it with your own audio files today!
```

## ⚡ Performance Tips

🚀 **Speed Optimization:**
- Use English-only models for English content
- Choose smaller models for faster processing
- Enable GPU acceleration if available

🎯 **Accuracy Tips:**
- Use higher quality audio recordings
- Minimize background noise
- Choose larger models for complex audio
- Adjust character limits based on your needs

## 🔧 Configuration

### Model Selection Guide
| Model Size | Speed | Accuracy | RAM Usage | Best For |
|------------|-------|----------|-----------|----------|
| Tiny | ⚡⚡⚡⚡ | ⭐⭐ | Low | Quick drafts |
| Base | ⚡⚡⚡ | ⭐⭐⭐ | Medium | **General use** |
| Small | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Quality balance |
| Medium | ⚡ | ⭐⭐⭐⭐⭐ | High | High accuracy |
| Large | 🐌 | ⭐⭐⭐⭐⭐ | Very High | Professional |

### Caption Formatting
- **20-30 chars**: Very short lines, frequent breaks
- **35-45 chars**: Standard subtitle length ⭐ **Recommended**
- **50-70 chars**: Longer lines, fewer breaks
- **80+ chars**: Very long lines, minimal breaks

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔀 Open a Pull Request

### Ideas for Contributions
- 🌐 Additional language support
- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 📱 Mobile responsiveness
- 🧪 Testing improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🤗 **Hugging Face** for the Transformers library
- 🎵 **OpenAI** for the Whisper models
- 🎨 **Streamlit** for the amazing web framework
- 📊 **librosa** for audio processing capabilities

## 📞 Support

Having issues? We're here to help!

- 🐛 **Bug Reports**: [Create an issue](https://github.com/yourusername/subtitles-generator/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/yourusername/subtitles-generator/discussions)
- 📧 **Direct Contact**: your.email@example.com

## 🔮 Roadmap

- [ ] 🎥 Video file support
- [ ] 🌐 Multi-language UI
- [ ] 📱 Mobile app version
- [ ] ☁️ Cloud deployment options
- [ ] 🔄 Batch processing
- [ ] 🎨 Custom styling for subtitles

---

<div align="center">

**Built with ❤️ by developers, for creators**

⭐ Star this repo if it helped you! ⭐

</div>