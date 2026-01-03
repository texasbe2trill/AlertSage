# Streamlit UI Guide

!!! tip "Launch the UI"
`bash
    streamlit run ui_premium.py
    `
The UI will open in your browser at `http://localhost:8501`

## Overview

The **NLP Cyber Incident Triage Laboratory** is a comprehensive Streamlit web application providing an interactive dashboard for security incident analysis. It combines real-time classification, visual analytics, threat intelligence, and SOC automation in a modern, user-friendly interface.

![Screenshot: UI Overview](images/ui-overview.png)

_Main interface showing the analysis dashboard_

---

## 🚀 Getting Started

### Launch the Application

From the project root directory:

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch UI
streamlit run ui_premium.py
```

The interface will automatically open in your default browser at `http://localhost:8501`.

### First-Time Setup

Before using the UI, ensure you have:

1. ✅ Installed all dependencies: `pip install -e .`
2. ✅ Model files downloaded (run `pytest tests/test_model.py -v` to trigger automatic download)
3. ✅ (Optional) LLM model for second opinions: Download Llama-2-7B-Chat GGUF

### Interactive Tutorial Walkthrough

When you first launch AlertSage, you'll see an **interactive tutorial** in the sidebar to help you get started:

#### 🎓 Tutorial Features

1. **Sample Incidents** - Try 3 pre-written security incidents:
   - **Suspicious Email with Attachment**: A phishing scenario with detailed indicators
   - **Multiple Failed Login Attempts**: Access abuse detection example
   - **Unusual Outbound Network Traffic**: Data exfiltration case study

2. **UI Mode Descriptions** - Learn about each analysis mode:
   - **Intelligence Dashboard**: Overview of security metrics and recent activity
   - **Single Incident Lab**: Analyze individual incidents in detail
   - **Advanced Search**: Query historical incident data
   - **Batch Analysis**: Process multiple incidents from files
   - **Bookmarks & History**: Access saved analyses
   - **Experimental Lab**: Advanced research features
   - **Settings & Profiles**: Customize preferences

3. **Configuration Help** - Understand key settings:
   - **Difficulty Level**: Controls classification strictness (Easy: 50%, Medium: 60%, Hard: 75%, Expert: 85%)
   - **Confidence Threshold**: Minimum score for certain predictions (recommended: 0.5-0.7)
   - **Max Classes**: Number of top predictions to display
   - **LLM Enhancement**: AI-powered second opinions for uncertain cases
   - **Advanced Visualizations**: Enhanced charts and risk assessments

4. **Tutorial Controls**:
   - ✅ **"Don't show this tutorial again"** checkbox to hide tutorial for returning users
   - 💡 **"Show Tutorial"** button to re-enable tutorial anytime

#### Using Sample Incidents

To try a sample incident:

1. Open the **"🎓 Getting Started Tutorial"** expander in the sidebar
2. Click any sample incident button (e.g., "Suspicious Email with Attachment")
3. Switch to **"Single Incident Lab"** mode
4. The sample incident text will be automatically loaded
5. Click **"Analyze"** to see the classification results

The tutorial is designed to work **without requiring any external data** - all samples are self-contained.

---

## 📋 Interface Modes

The UI offers seven primary analysis modes accessible from the sidebar:

### 🔍 Single Incident Analysis

Analyze individual security incidents with comprehensive intelligence.

![Screenshot: Single Incident Mode](images/ui-single-incident.png)

_Single incident analysis with real-time classification_

**Features:**

- Real-time incident classification
- Confidence scoring with visual gauge
- Probability distribution charts
- MITRE ATT&CK technique mapping
- Threat intelligence panel
- SOC playbook recommendations
- Risk radar visualization

**Workflow:**

1. Enter incident description in the text area
2. Configure analysis settings in sidebar (threshold, difficulty, LLM)
3. Click "🔍 Analyze Incident"
4. Explore results across five analysis tabs

### 📊 Bulk Analysis Intelligence Center

Process multiple incidents from uploaded files with advanced analytics.

![Screenshot: Bulk Analysis Mode](images/ui-bulk-analysis.png)

_Bulk processing dashboard with aggregate metrics_

**Features:**

- CSV/TXT file upload support
- Batch processing with progress tracking
- Aggregate statistics and metrics
- LLM upgrade tracking (shows when AI changes classifications)
- Interactive filtering by label, confidence, uncertainty
- Export results as CSV/JSON
- Comprehensive threat intelligence briefs

**Workflow:**

1. Upload incidents file (CSV with `description` column or TXT with one incident per line)
2. Configure batch processing settings
3. Monitor real-time progress
4. Review aggregate analytics
5. Filter and export results

### 🧪 Experimental Lab

Advanced features for research and experimentation.

![Screenshot: Experimental Lab](images/ui-experimental.png)
_Experimental analysis tools_

**Features:**

- Text similarity analysis
- Incident clustering
- Model performance comparison
- Synthetic data generation
- Advanced feature extraction
- IOC lookup and threat feeds

---

## 🎛️ Sidebar Configuration

The sidebar provides comprehensive control over analysis parameters:

![Screenshot: Sidebar Settings](images/ui-sidebar.png)

_Configuration panel with all analysis settings_

### Analysis Settings

**Difficulty Mode**

- `default` - Standard thresholds (50% confidence)
- `soc-medium` - Moderate strictness (60% confidence)
- `soc-hard` - Maximum strictness (75% confidence)

**Confidence Threshold**

- Slider: 0.0 to 1.0
- Default: 0.50
- Controls when predictions are marked "uncertain"

**Max Classes**

- Number of top predictions to display
- Range: 1-7
- Useful for exploring runner-up classifications

### LLM Configuration

**Enable LLM Second Opinion**

- Toggle AI-assisted classification
- Engages automatically for uncertain cases
- Provides alternative perspective with rationale

**Debug Mode**

- Shows detailed LLM prompts and responses
- Useful for troubleshooting
- Performance analysis

### Visualization Options

**Advanced Visualizations**

- Enable enhanced charts and graphs
- Risk radar charts
- Confidence distributions
- MITRE technique heatmaps

---

## 📊 Analysis Tabs (Single Incident)

### Tab 1: 🎯 Analysis

Core classification results with key metrics.

![Screenshot: Analysis Tab](images/ui-tab-analysis.png)
_Main analysis results with confidence metrics_

**Displays:**

- Final classification label
- Confidence score with gauge visualization
- Uncertainty level indicator
- Top-N probability distribution
- Class probability pie chart

### Tab 2: 📊 Visualizations

Interactive charts and visual analytics.

![Screenshot: Visualizations Tab](images/ui-tab-visualizations.png)
_Visual analytics dashboard_

**Charts:**

- Confidence gauge (speedometer-style)
- Probability distribution (pie chart)
- Risk radar (multi-dimensional assessment)
- Text complexity metrics

### Tab 3: 🕵️ Threat Intel

Comprehensive threat intelligence analysis.

![Screenshot: Threat Intel Tab](images/ui-tab-threat-intel.png)
_Threat intelligence panel with IOC extraction_

**Features:**

- MITRE ATT&CK technique mapping
- IOC extraction (IPs, URLs, emails)
- Attack sophistication scoring
- Threat landscape context
- Related campaigns/TTPs

### Tab 4: 📋 SOC Playbook

Context-aware incident response recommendations.

![Screenshot: Playbook Tab](images/ui-tab-playbook.png)
_SOC playbook with actionable recommendations_

**Provides:**

- Incident priority (P1-P5)
- Response timeline
- Step-by-step actions
- Context-specific guidance
- Escalation paths

### Tab 5: 🔧 Technical Details

Raw data and technical information.

![Screenshot: Technical Details Tab](images/ui-tab-technical.png)
_Technical debugging and raw JSON output_

**Includes:**

- Full JSON response
- Text complexity analysis
- Model metadata
- Debug information
- LLM prompts/responses (if enabled)

---

## 📈 Bulk Analysis Features

### Upload & Processing

**Supported Formats:**

**CSV:**

```csv
description
"User reported suspicious email with attachment"
"Multiple failed login attempts from Asia"
"Unusual outbound traffic to 192.168.1.100"
```

**TXT (one incident per line):**

```
User reported suspicious email with attachment
Multiple failed login attempts from Asia
Unusual outbound traffic to 192.168.1.100
```

**Processing:**

- Real-time progress bar
- Estimated time remaining
- Incident counter
- Error handling with retry logic

![Screenshot: Bulk Upload](images/ui-bulk-upload.png)
_File upload and processing interface_

### Results Dashboard

After processing completes, view comprehensive analytics:

![Screenshot: Bulk Results](images/ui-bulk-results.png)
_Aggregate results with filtering and export_

**Metrics:**

- Total incidents processed
- Label distribution
- Average confidence
- LLM-resolved count (when second opinion used)
- Uncertain case count

**Interactive Table:**

- Sortable columns
- Filterable by label, confidence, uncertainty
- Expandable rows for full incident text
- Color-coded by risk level

### Advanced Analytics

Access four analytics dashboards:

**📈 Overview**

- Label distribution pie chart
- Confidence histogram
- Timeline analysis
- MITRE technique frequency

**🎯 Confidence Analysis**

- Confidence vs label scatter plot
- Uncertainty distribution
- High/low confidence breakdown
- Threshold impact analysis

**⚡ Performance**

- Processing speed metrics
- Model inference time
- LLM overhead analysis
- Resource utilization

**🔬 Deep Dive**

- Text complexity analysis
- N-gram frequency
- IOC extraction summary
- Correlation matrices

![Screenshot: Advanced Analytics](images/ui-bulk-analytics.png)
_Advanced analytics with multiple visualization panels_

### Export Options

Download results in multiple formats:

**CSV Export:**

```csv
description,label,confidence,display_label,llm_override,mitre_techniques
"...",phishing,0.87,phishing,No,"T1566.001,T1204.002"
```

**JSON Export:**

```json
{
  "description": "...",
  "label": "phishing",
  "confidence": 0.87,
  "display_label": "phishing",
  "llm_second_opinion": {...},
  "probabilities": {...},
  "mitre_techniques": ["T1566.001"]
}
```

**Threat Intelligence Brief:**

- Executive summary (Markdown/PDF)
- MITRE coverage report
- Critical incidents highlight
- Strategic recommendations

---

## 🧪 Experimental Lab Tools

### Text Similarity Analysis

Compare incidents and find similar patterns.

![Screenshot: Similarity Analysis](images/ui-similarity.png)
_Text similarity clustering visualization_

**Methods:**

- TF-IDF cosine similarity
- Semantic embeddings
- Clustering (K-means, DBSCAN)
- Similarity heatmaps

### Model Comparison

Benchmark different classifiers.

**Models:**

- Logistic Regression (baseline)
- Random Forest
- Linear SVM
- Ensemble methods

**Metrics:**

- Accuracy comparison
- Confusion matrices
- Per-class performance
- Feature importance

### Synthetic Data Generation

Create test datasets on-demand.

**Parameters:**

- Incident type
- Complexity level
- Batch size
- Include IOCs/MITRE/timestamps
- Export format

---

## ⚙️ Configuration

### Environment Variables

The UI respects the same environment variables as the CLI:

```bash
# LLM Configuration
export TRIAGE_LLM_MODEL=/path/to/llama-2-7b-chat.Q5_K_S.gguf
export TRIAGE_LLM_DEBUG=1
export NLP_TRIAGE_LLM_TEMPERATURE=0.2
export NLP_TRIAGE_LLM_MAX_TOKENS=512

# Model Paths
export TRIAGE_LLM_CTX=4096
```

See [Configuration Guide](configuration.md) for complete settings.

### Custom Styling

Modify `ui_premium.py` to customize the interface:

**Color Schemes:**

```python
# Located at top of ui_premium.py
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    ...
}
```

**Chart Types:**

- Plotly graph configurations
- Streamlit theme settings
- Layout adjustments

---

## 💡 Tips & Best Practices

### Performance Optimization

✅ **Use baseline first** - Test without LLM for 10-20x faster processing  
✅ **Batch processing** - Process multiple incidents in bulk mode  
✅ **Adjust max classes** - Limit to 3-5 for faster rendering  
✅ **Cache results** - Export and reload rather than re-analyzing

### Accuracy Improvements

✅ **Tune thresholds** - Lower (0.3-0.4) for coverage, higher (0.6-0.7) for precision  
✅ **Use difficulty modes** - `soc-hard` for critical infrastructure  
✅ **Enable LLM selectively** - Only for uncertain/high-stakes cases  
✅ **Review uncertain** - Manual analysis for low-confidence predictions

### Workflow Recommendations

✅ **Single → Bulk** - Test single incidents first, then scale to bulk  
✅ **Export everything** - Save results for audit trails  
✅ **Use playbooks** - Follow SOC recommendations systematically  
✅ **Monitor metrics** - Track confidence trends over time

### What to Avoid

❌ **Don't trust blindly** - Always review uncertain predictions  
❌ **Don't over-rely on LLM** - It's decision support, not ground truth  
❌ **Don't ignore confidence** - Low scores = unreliable classifications  
❌ **Don't skip validation** - Verify results against ground truth when available

---

## 🔧 Troubleshooting

### Common Issues

**"Model files not found"**

```bash
# Trigger automatic download
pytest tests/test_model.py -v
```

**Slow LLM processing**

- Use quantized models (Q5_K_S recommended)
- Enable GPU acceleration via llama-cpp-python
- Reduce context window: `export TRIAGE_LLM_CTX=2048`
- Lower temperature for faster generation

**CSV upload fails**

- Ensure `description` column exists
- Check UTF-8 encoding
- Remove empty rows
- Verify proper CSV delimiter (comma)

**UI crashes or freezes**

- Check terminal output for errors
- Verify sufficient RAM (8GB+ recommended)
- Close other applications
- Reduce LLM context window if OOM

**Blank visualizations**

- Enable "Advanced Visualizations" in sidebar
- Check browser console for JavaScript errors
- Refresh page
- Try different browser (Chrome/Firefox recommended)

### Debug Mode

Enable detailed logging:

```bash
export TRIAGE_LLM_DEBUG=1
streamlit run ui_premium.py
```

Check terminal output for:

- LLM prompts and responses
- Model loading status
- Processing errors
- Performance metrics

---

## 🔗 Related Documentation

- [CLI Usage](cli.md) - Command-line interface guide
- [LLM Integration](llm-integration.md) - Setting up local LLM models
- [Configuration](configuration.md) - Environment variables and settings
- [API Reference](api-reference.md) - Programmatic access
- [Architecture](architecture.md) - System design and components

---

**Need help?** Open an issue on [GitHub](https://github.com/texasbe2trill/AlertSage/issues) or check the [FAQ](faq.md).
