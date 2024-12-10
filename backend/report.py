import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

def process_report():

    csv_file = "temp_files/data.csv"  
    data = pd.read_csv(csv_file)

    columns_to_display = [
        'Timestamp', 
        'Joint Angles', 
        'Detected Injuries', 
        'Risk Level', 
        'Overall Assessment'
    ]

    try:
        data['Risk Level'] = pd.to_numeric(data['Risk Level'], errors='coerce')
        
        high_risk_data = data[data['Risk Level'] > 12][columns_to_display]
        
        if high_risk_data.empty:
            print("Warning: No entries found with Risk Level > 12")
            display_data = data[columns_to_display]
        else:
            display_data = high_risk_data
            print(f"Found {len(high_risk_data)} high-risk entries")

    except KeyError as e:
        print(f"Error: Missing column in the CSV. {e}")
        print("Available columns in the CSV:", list(data.columns))
        display_data = data

    class PDF(FPDF):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            montserrat = 'Montserrat-Black.ttf'
            
            self.add_font('Montserrat', '', montserrat, uni=True)

        def header(self):

            self.set_fill_color(16, 17, 36)
            self.image("wave_headertest.png", 0, 0, 210, 0, 'PNG')

            self.set_text_color(255, 255, 255)
            self.set_font('Montserrat', '', 21)
            self.set_xy(158, 8)
            self.cell(0, 10, 'GestureAI', 0, 0)


            self.set_y(40)

        def footer(self):
            # Page number in footer
            self.set_y(-15)
            self.set_font('Montserrat', '', 10)
            self.cell(0, 10, f' {self.page_no()}', 0, 0, 'C')

        def add_title(self, title):
            self.set_font("Arial", "B", 18)
            self.cell(0, 10, title, 0, 1, "C")
            self.ln(5)

        def add_table(self, data):
            # Table column widths
            col_widths = [30, 40, 40, 30, 40]
            
            # Table header
            self.set_font('Arial', 'B', 10)
            self.set_fill_color(200, 220, 255)
            self.set_x(15)
            for i, col in enumerate(data.columns):
                self.cell(col_widths[i], 15, str(col), 1, 0, 'C', fill=True)
            self.ln()
            
            # Table rows
            self.set_font('Arial', '', 9)
            for index, row in data.iterrows():
                # Reset x position to start of page
                self.set_x(15)
                
                # Truncate or wrap long text
                row_data = [
                    str(row[col])[:20] + ('...' if len(str(row[col])) > 30 else '') 
                    for col in data.columns
                ]
                
                # Draw cells for this row
                for i, cell_value in enumerate(row_data):
                    self.cell(col_widths[i], 10, cell_value, 1)
                self.ln()
    # Step 4: Create the PDF
    pdf = PDF()
    pdf.add_page()

    # Add a title
    pdf.add_title("Report After Processing The Video Data")

    # Add data table
    pdf.add_table(display_data)

    # Save the PDF
    output_pdf = "temp_files/report.pdf"
    pdf.output(output_pdf)    

    print(f"PDF report saved as {output_pdf}")

if __name__ == "__main__":

    process_report()
