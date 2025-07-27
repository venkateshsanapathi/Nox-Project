"""
Advanced NOx Conversion Efficiency Prediction GUI
Simple Tkinter interface for NOx prediction with only important parameters.
Run with: python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
from pathlib import Path
import pandas as pd # Added for reading feature_importance.csv

# Add the project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import predictor (check if available)
try:
    from src.predictor import predict_nox_efficiency, get_feature_info
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)
except Exception as e: # Catch errors during initial model loading within predictor
    MODEL_AVAILABLE = False
    IMPORT_ERROR = f"Error loading model artifacts: {e}. Ensure training completed successfully."

class NOxPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("NOx Conversion Efficiency Predictor v2.0")
        self.geometry("600x700")
        self.resizable(True, True)

        # Configure style
        self.configure(bg='#f0f0f0')
        style = ttk.Style()
        style.theme_use('clam')

        # Check if model is available
        if not MODEL_AVAILABLE:
            self.show_model_error()
            return

        self.setup_ui()

    def show_model_error(self):
        """Show error if model is not available"""
        error_frame = ttk.Frame(self)
        error_frame.pack(expand=True, fill='both', padx=20, pady=20)

        ttk.Label(error_frame, text="Model Not Available", 
                 font=("Arial", 16, "bold"), foreground="red").pack(pady=10)

        error_msg = f"""
Model artifacts not found. Please run training first:

1. Ensure 'nox_data.xlsx' is in the same directory
2. Run: python train_model.py
3. Wait for training to complete
4. Then run this GUI again

Error details: {IMPORT_ERROR if 'IMPORT_ERROR' in globals() else 'Unknown error'}
        """

        ttk.Label(error_frame, text=error_msg, font=("Arial", 10), 
                 justify="left", wraplength=500).pack(pady=10)

        ttk.Button(error_frame, text="Exit", command=self.quit).pack(pady=10)

    def setup_ui(self):
        """Setup the main user interface"""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=20, pady=10)

        ttk.Label(header_frame, text="NOx Conversion Efficiency Predictor", 
                 font=("Arial", 18, "bold")).pack()
        ttk.Label(header_frame, text="Enter engine and system parameters for SCR efficiency prediction", 
                 font=("Arial", 10), foreground="gray").pack(pady=5)

        # Main input frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Canvas for scrolling
        canvas = tk.Canvas(main_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input fields
        self.entries = {}
        self.create_input_fields(scrollable_frame) # This is where the input fields are actually created

        # Control buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(fill='x', padx=20, pady=10)

        ttk.Button(control_frame, text="Load Defaults", 
                  command=self.load_defaults).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", 
                  command=self.clear_all).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Predict NOx CE", 
                  command=self.predict_threaded, 
                  style="Accent.TButton").pack(side='right', padx=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(self, text="Prediction Results", padding=15)
        self.results_frame.pack(fill='x', padx=20, pady=10)

        self.result_text = tk.Text(self.results_frame, height=8, wrap='word', 
                                  font=("Consolas", 10), state='disabled')
        self.result_text.pack(fill='x')

        # Status bar
        self.status_var = tk.StringVar(value="Ready for prediction")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief='sunken', 
                              font=("Arial", 9))
        status_bar.pack(fill='x', side='bottom')

    def get_top_n_features(self, n=10):
        """Loads feature importances and returns the top N feature names."""
        try:
            importance_df = pd.read_csv(PROJECT_ROOT / "models" / "feature_importance.csv")
            top_features = importance_df['feature'].head(n).tolist()
            return top_features
        except FileNotFoundError:
            print("Warning: feature_importance.csv not found. Using all features for GUI.")
            return self.all_features # Fallback to all features
        except Exception as e:
            print(f"Error loading feature importances: {e}. Using all features for GUI.")
            return self.all_features

    def create_input_fields(self, parent):
        """Create input fields for all required features"""
        feature_info = get_feature_info()
        self.all_features = feature_info['features']
        self.feature_descriptions = feature_info['descriptions']
        self.feature_ranges = feature_info['ranges']

        # Determine which features to display
        self.display_features = self.get_top_n_features(n=10) # Display top 10 features
        
        # Identify features that are part of the model but not displayed in GUI
        self.hidden_features = [f for f in self.all_features if f not in self.display_features]

        # Input fields for displayed features
        displayed_features_frame = ttk.LabelFrame(parent, text="Important Input Parameters", padding=10)
        displayed_features_frame.pack(fill='x', pady=5)

        for feature in self.display_features:
            self.create_feature_input(displayed_features_frame, feature, self.feature_descriptions, self.feature_ranges)

        # Inform the user about hidden features if any
        if self.hidden_features:
            ttk.Label(parent, text=f"Note: {len(self.hidden_features)} additional parameters will be set to default values for prediction.",
                      font=("Arial", 9), foreground="gray").pack(pady=5)

    def create_feature_input(self, parent, feature, descriptions, ranges):
        """Create input field for a single feature"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=3)

        # Label with description
        desc = descriptions.get(feature, feature)
        range_info = ranges.get(feature, (0, 1000))
        label_text = f"{desc}:"

        label = ttk.Label(frame, text=label_text, width=25, anchor='w')
        label.pack(side='left')

        # Entry field
        entry = ttk.Entry(frame, width=15, font=("Arial", 10))
        entry.pack(side='left', padx=5)

        # Range info
        range_label = ttk.Label(frame, text=f"({range_info[0]:.8f} - {range_info[1]:.8f})", 
                               font=("Arial", 8), foreground="gray")
        range_label.pack(side='left', padx=5)

        self.entries[feature] = entry

        # Bind validation
        entry.bind('<FocusOut>', lambda e, f=feature: self.validate_field(f))

    def validate_field(self, feature):
        """Validate individual field"""
        try:
            value = self.entries[feature].get()
            if value:
                float_val = float(value)
                range_info = self.feature_ranges.get(feature, (None, None))
                if range_info[0] is not None and range_info[1] is not None:
                    if not (range_info[0] <= float_val <= range_info[1]):
                        self.entries[feature].configure(style="Error.TEntry")
                        return False
                else: # No numerical range defined, still ensure it's a number
                    pass
                self.entries[feature].configure(style="TEntry")
                return True
        except ValueError:
            self.entries[feature].configure(style="Error.TEntry")
            return False
        return True

    def load_defaults(self):
        """Load default values for displayed fields and populate hidden fields."""
        for feature in self.display_features: # Only iterate over displayed features
            if feature in self.entries:
                range_info = self.feature_ranges.get(feature, (None, None))
                if range_info[0] is not None and range_info[1] is not None:
                    default_value = (range_info[0] + range_info[1]) / 2
                    if feature == 'V_SCR_Ct_NH3_Slip_Detected':
                        default_value = 0
                    elif feature == 'V_ATM_f_g_HC_fb':
                        default_value = 0
                    self.entries[feature].delete(0, tk.END)
                    self.entries[feature].insert(0, str(default_value)) # Insert full precision string
                else:
                    self.entries[feature].delete(0, tk.END)
                    self.entries[feature].insert(0, "0") # Default to 0 for unknown numeric ranges

        self.status_var.set("Default values loaded for displayed fields.")

    def clear_all(self):
        """Clear all input fields"""
        for entry in self.entries.values(): # Only iterate over displayed features
            entry.delete(0, tk.END)
            entry.configure(style="TEntry")
        self.update_results("", clear=True)
        self.status_var.set("All displayed fields cleared.")

    def predict_threaded(self):
        """Run prediction in separate thread to avoid GUI freeze"""
        threading.Thread(target=self.predict, daemon=True).start()

    def predict(self):
        """Make NOx conversion efficiency prediction"""
        try:
            self.status_var.set("Making prediction...")

            # Collect input values for displayed features
            user_inputs = {}
            for feature in self.display_features:
                value = self.entries[feature].get()
                if not value:
                    raise ValueError(f"Please enter value for {self.feature_descriptions.get(feature, feature)}")
                user_inputs[feature] = value

            # Automatically set values for hidden features
            for feature in self.hidden_features:
                range_info = self.feature_ranges.get(feature, (None, None))
                if range_info[0] is not None and range_info[1] is not None:
                    # Set to midpoint for hidden numerical features
                    default_value = (range_info[0] + range_info[1]) / 2
                    if feature == 'V_SCR_Ct_NH3_Slip_Detected':
                        default_value = 0
                    elif feature == 'V_ATM_f_g_HC_fb':
                        default_value = 0
                    user_inputs[feature] = str(default_value) # Insert full precision string
                else:
                    user_inputs[feature] = "0" # Default to 0 for unknown numeric ranges
                    
            # Make prediction
            result = predict_nox_efficiency(user_inputs)

            # Display results
            self.display_results(result, user_inputs)
            self.status_var.set("Prediction completed successfully")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_var.set("Prediction failed")

    def display_results(self, result, inputs):
        """Display prediction results"""
        nox_ce = result['nox_ce']
        category = result['category']
        confidence = result['confidence']

        # Format results for displayed features only
        results_text = f"""
NOx CONVERSION EFFICIENCY PREDICTION
{'='*50}

RESULTS:
   NOx Conversion Efficiency: {nox_ce}%
   Performance Category:      {category}
   Prediction Confidence:     {confidence}%

INTERPRETATION:
   {self.get_interpretation(nox_ce)}

INPUT VALUES (Displayed):
"""

        for feature in self.display_features: # Only show displayed features in result summary
            value = inputs.get(feature, "N/A")
            desc = self.feature_descriptions.get(feature, feature)
            results_text += f"   {desc:<25}: {value}\n"
        
        if self.hidden_features:
            results_text += f"\nINPUT VALUES (Automatically Set for {len(self.hidden_features)} Hidden Parameters):\n"
            for feature in self.hidden_features:
                value = inputs.get(feature, "N/A")
                desc = self.feature_descriptions.get(feature, feature)
                results_text += f"   {desc:<25}: {value}\n"

        results_text += f"""
RECOMMENDATIONS:
   {self.get_recommendations(nox_ce)}
"""
        self.update_results(results_text)


    def get_interpretation(self, nox_ce):
        """Get interpretation of NOx conversion efficiency"""
        if nox_ce >= 95:
            return "Excellent SCR performance. System operating optimally."
        elif nox_ce >= 90:
            return "Good SCR performance. Minor optimization possible."
        elif nox_ce >= 80:
            return "Fair SCR performance. Consider system adjustments."
        else:
            return "Poor SCR performance. Immediate attention required."

    def get_recommendations(self, nox_ce):
        """Get recommendations for improving NOx conversion efficiency (simplified)"""
        recommendations = []

        if nox_ce < 80:
            recommendations.append("  Check overall engine health and emission system components.")
            recommendations.append("  Review all input parameters for out-of-range values.")
        elif nox_ce < 90:
            recommendations.append("  Small adjustments to key parameters may improve efficiency.")
            recommendations.append("  Consult data sheets for optimal operating points for specific sensors/actuators.")
        else:
            recommendations.append("  System is performing well. Maintain current operational practices.")
            
        return "\n   ".join(recommendations) if recommendations else "System appears well-optimized."

    def update_results(self, text, clear=False):
        """Update results display"""
        self.result_text.configure(state='normal')
        if clear:
            self.result_text.delete(1.0, tk.END)
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, text)
        self.result_text.configure(state='disabled')

def main():
    """Main application entry point"""
    app = NOxPredictionApp()

    # Configure error style
    style = ttk.Style()
    style.configure("Error.TEntry", fieldbackground="lightcoral")
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
