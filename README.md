# Research Paper Tagger

An intelligent Python tool that automatically assigns category tags to research papers based on their content. The tool uses advanced natural language processing techniques to analyze PDF documents and classify them into relevant research categories.

## Features

### Core Functionality
- **Automatic Tagging**: No manual categorization needed - the tool analyzes paper content and assigns relevant tags
- **Multi-tag Support**: Papers can be assigned multiple topic tags based on their content
- **Batch Processing**: Process entire folders of PDFs at once
- **Offline Operation**: Uses local NLP models - no internet connection required after initial setup
- **Custom Categories**: Define your own tag categories and keywords

### PDF Processing
- **Multiple Extraction Methods**: Supports PyMuPDF, pdfplumber, and PyPDF2 for robust text extraction
- **Intelligent Section Parsing**: Automatically extracts abstracts, keywords, introduction, and conclusion
- **Metadata Extraction**: Retrieves PDF metadata when available

### Output Options
- **CSV Export**: Export results to CSV for analysis and spreadsheet software
- **JSON Export**: Machine-readable JSON format for integration with other tools
- **File Organization**: Automatically organize papers into tag-based directory structures
- **Filename Tagging**: Add tags directly to filenames for easy identification
- **Summary Reports**: Generate detailed statistics and tag distribution reports

### Customization
- **Configurable Categories**: Add custom research categories with specific keywords
- **Adjustable Thresholds**: Fine-tune confidence levels for tag assignment
- **Flexible Settings**: Customize processing parameters via YAML configuration files

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sreeram5678/research-paper-tagger.git
   cd research-paper-tagger
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (done automatically on first run):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Quick Start

### Tag a Single Paper
```bash
python main.py tag-single path/to/paper.pdf
```

### Tag Multiple Papers
```bash
python main.py tag-batch path/to/pdf/directory/
```

### With Custom Settings
```bash
python main.py tag-batch papers/ --threshold 0.4 --max-tags 3 --format both --organize
```

## Usage Examples

### Basic Usage

**Tag a single PDF**:
```bash
python main.py tag-single research_paper.pdf
```

**Process all PDFs in a directory**:
```bash
python main.py tag-batch papers/ --summary
```

### Advanced Usage

**Use custom configuration**:
```bash
python main.py tag-batch papers/ --config custom_config.yaml --organize --rename
```

**Export in multiple formats with file organization**:
```bash
python main.py tag-batch papers/ --format both --organize --threshold 0.3 --max-tags 5
```

**Test PDF extraction**:
```bash
python main.py test-extraction paper.pdf --method pymupdf
```

## Command Reference

### Main Commands

- `tag-single FILE`: Tag a single PDF file
- `tag-batch DIRECTORY`: Tag all PDFs in a directory
- `create-config`: Create a sample configuration file
- `validate-config FILE`: Validate a configuration file
- `list-categories`: Show available categories and keywords
- `test-extraction FILE`: Test PDF text extraction

### Common Options

- `--config, -c`: Custom configuration file
- `--output, -o`: Output directory
- `--format`: Output format (csv, json, both)
- `--threshold, -t`: Confidence threshold (0.0-1.0)
- `--max-tags`: Maximum tags per paper
- `--organize`: Organize files by tags
- `--rename`: Rename files with tags
- `--summary`: Create summary report
- `--verbose, -v`: Enable verbose logging

## Configuration

### Creating a Custom Configuration

```bash
python main.py create-config --output my_config.yaml
```

### Sample Configuration

```yaml
# Processing settings
processing:
  confidence_threshold: 0.3
  max_tags: 5
  extraction_method: "auto"

# Custom categories
categories:
  "Quantum Computing":
    - "quantum computing"
    - "quantum algorithm"
    - "qubit"
    - "quantum circuit"
  
  "Blockchain":
    - "blockchain"
    - "cryptocurrency"
    - "smart contract"
    - "distributed ledger"

# Output settings
output:
  default_format: "csv"
  create_summary: true
  organize_by_tags: false
```

## Built-in Categories

The tool comes with comprehensive categories for common research areas:

- **Natural Language Processing**: NLP, text mining, language models, transformers
- **Computer Vision**: Image processing, object detection, CNNs, visual recognition
- **Machine Learning**: Supervised/unsupervised learning, neural networks, algorithms
- **Deep Learning**: Deep neural networks, backpropagation, optimization
- **Reinforcement Learning**: Q-learning, policy gradients, MDPs
- **Optimization**: Linear programming, metaheuristics, gradient methods
- **Data Mining**: Knowledge discovery, clustering, recommendation systems
- **Graph Neural Networks**: GNNs, graph classification, network analysis
- **Meta-Learning**: Few-shot learning, transfer learning, MAML
- **Federated Learning**: Distributed learning, privacy-preserving ML

And many more! Use `python main.py list-categories` to see all available categories.

## Project Structure

```
research-paper-tagger/
├── src/                          # Source code
│   ├── __init__.py
│   ├── pdf_extractor.py         # PDF text extraction
│   ├── nlp_classifier.py        # NLP classification engine
│   ├── paper_tagger.py          # Main tagging logic
│   ├── output_handlers.py       # Export and organization
│   └── config_loader.py         # Configuration management
├── config/                       # Configuration files
│   └── default_config.yaml      # Default configuration
├── data/                         # Input PDFs (add your papers here)
├── output/                       # Generated results
├── tests/                        # Test files
├── main.py                       # Command-line interface
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Output Examples

### CSV Output
```csv
filename,success,tags,tag_scores,processing_time
paper1.pdf,True,"Deep Learning; Computer Vision","Deep Learning:0.856; Computer Vision:0.743",2.34
paper2.pdf,True,"NLP; Machine Learning","NLP:0.923; Machine Learning:0.651",1.87
```

### Summary Report
```
RESEARCH PAPER TAGGING SUMMARY
==================================================

Total papers processed: 25
Successfully tagged: 24
Failed to process: 1
Success rate: 96.0%

Average tags per paper: 2.3
Total unique tags: 8

TAG FREQUENCY DISTRIBUTION:
------------------------------
Deep Learning              12 papers (0.756 avg score)
Machine Learning           10 papers (0.698 avg score)
Computer Vision             8 papers (0.723 avg score)
```

## Troubleshooting

### Common Issues

**PDF extraction fails**:
- Try different extraction methods: `--extraction-method pdfplumber`
- Some PDFs may be scanned images - consider OCR preprocessing

**Low tagging accuracy**:
- Adjust confidence threshold: `--threshold 0.2`
- Add custom categories for your specific domain
- Check if paper abstracts are being extracted properly

**Memory issues with large batches**:
- Process smaller batches
- Reduce TF-IDF features in advanced configuration

### Getting Help

- Use `--verbose` flag for detailed logging
- Test extraction with `test-extraction` command
- Validate configuration with `validate-config`

## Contributing

We welcome contributions! Here are some ways you can help:

### Adding New Categories

1. Edit your configuration file:
```yaml
categories:
  "Your New Category":
    - "keyword1"
    - "keyword2"
    - "specific term"
```

2. Test with your papers:
```bash
python main.py tag-single test_paper.pdf --config your_config.yaml
```

### Improving Extraction

The tool uses multiple PDF extraction libraries for robustness. If you encounter PDFs that don't extract well, you can:

1. Try different extraction methods
2. Preprocess PDFs with OCR tools
3. Contribute improvements to the extraction pipeline

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Testing

Run the test suite to ensure everything works correctly:

```bash
python -m pytest tests/
```

## License

This project is proprietary software owned by Sreeram Lagisetty. 
All rights reserved. Commercial use requires explicit written permission.

## Acknowledgments

- Built with scikit-learn for machine learning
- Uses NLTK for natural language processing
- PDF processing with PyMuPDF, pdfplumber, and PyPDF2
- Command-line interface powered by Click
- Progress tracking with tqdm

## Author

**Sreeram Lagisetty**

- **Email**: [sreeram.lagisetty@gmail.com](mailto:sreeram.lagisetty@gmail.com)
- **GitHub**: [Sreeram 5678](https://github.com/Sreeram5678)
- **Instagram**: [@sreeram_3012](https://www.instagram.com/sreeram_3012?igsh=N2Fub3A5eWF4cjJs&utm_source=qr)

## Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the command reference
3. Use verbose logging to debug issues
4. Open an issue on GitHub for bugs or feature requests
5. Contact me directly via email or social media

---

**Star this repository if you find it helpful!** ⭐
