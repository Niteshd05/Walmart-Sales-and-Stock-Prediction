# 🛒 Walmart Demand Prediction & Inventory Optimization System

An intelligent demand forecasting and inventory management system built for Walmart retail operations. This project uses machine learning to predict product demand and optimize stock levels while minimizing waste.

## 🎯 Features

- **Demand Prediction**: Uses Random Forest regression to forecast daily demand for different product categories
- **Inventory Optimization**: Calculates optimal stock levels, reorder points, and safety stock
- **Waste Reduction**: Provides shelf-life-based stock recommendations to minimize product expiration
- **Interactive Dashboard**: Beautiful Streamlit web interface with real-time predictions and visualizations
- **Multi-Category Support**: Handles 6 product categories (Health & Beauty, Electronics, Home & Lifestyle, Sports & Travel, Food & Beverages, Fashion Accessories)

## 📊 Dashboard Preview

The Streamlit interface provides:
- 📈 Real-time demand predictions
- 📊 Interactive charts and visualizations
- 📋 Detailed stock recommendations
- 📥 Export functionality for predictions
- 🔄 Dynamic product category selection

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Sales data in CSV format

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/walmart-demand-prediction.git
cd walmart-demand-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python walmart_model_training.py
```
*Enter the path to your sales.csv file when prompted*

4. **Launch the dashboard**
```bash
streamlit run streamlit_app.py
```

5. **Access the web interface**
Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
walmart-demand-prediction/
├── walmart_model_training.py    # Model training script
├── streamlit_app.py            # Streamlit web interface
├── requirements.txt            # Python dependencies
├── walmart_model.pkl          # Trained model (generated after training)
└── README.md                  # This file
```

## 🎛️ Usage

### Training the Model
1. Prepare your sales data in CSV format with columns: Date, Time, Product line, Quantity, etc.
2. Run the training script and provide the path to your CSV file
3. The script will preprocess data, train the model, and save it as `walmart_model.pkl`

### Using the Dashboard
1. Launch the Streamlit app
2. Load the trained model using the sidebar
3. Select a product category
4. Choose prediction period (7-90 days)
5. Click "Generate Prediction" to get results
6. View recommendations and export results

## 📈 Model Performance

The system uses Random Forest Regression with the following features:
- **Temporal Features**: Month, Day, Day of Week, Hour
- **Product Features**: Category, Price Range, Historical Sales
- **Customer Features**: Customer Type, Gender, Payment Method
- **Location Features**: Branch, City

Performance metrics are displayed during training, including:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Feature Importance Analysis

## 🛍️ Supported Product Categories

1. **Health and Beauty** (365-day shelf life)
2. **Electronic Accessories** (730-day shelf life)
3. **Home and Lifestyle** (1095-day shelf life)
4. **Sports and Travel** (730-day shelf life)
5. **Food and Beverages** (30-day shelf life)
6. **Fashion Accessories** (365-day shelf life)

## 📊 Key Metrics Provided

### Demand Forecasting
- Predicted daily demand
- Historical demand statistics
- Demand trend analysis
- Seasonal patterns

### Inventory Optimization
- **Safety Stock**: Buffer stock to prevent stockouts
- **Reorder Point**: When to place new orders
- **Optimal Stock Level**: Ideal inventory quantity
- **Waste Optimized Stock**: Shelf-life conscious recommendations

### Business Intelligence
- Restock timing recommendations
- Lead time considerations
- Cost optimization suggestions
- Waste reduction strategies

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations

### Data Processing
- Automatic date/time parsing
- Feature engineering for temporal patterns
- Categorical encoding for product attributes
- Data validation and cleaning

### Model Architecture
- Random Forest Regressor (100 estimators)
- Cross-validation for model selection
- Feature importance analysis
- Hyperparameter optimization

## 📝 Data Format

Your sales CSV should include these columns:
- `Date`: Transaction date
- `Time`: Transaction time
- `Product line`: Product category
- `Quantity`: Units sold
- `Unit price`: Price per unit
- `Branch`: Store branch
- `City`: Store location
- `Customer type`: Customer segment
- `Gender`: Customer gender
- `Payment`: Payment method
- `Rating`: Customer rating

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for Walmart retail operations optimization
- Inspired by modern supply chain management practices
- Uses industry-standard ML techniques for demand forecasting

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**⭐ Star this repository if you find it helpful!**
