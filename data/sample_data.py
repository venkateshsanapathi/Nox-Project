import pandas as pd
import numpy as np

def get_sample_data(num_rows=100):
    """
    Provides a sample dataset transcribed from the provided images and expanded.
    """
    # Base data from Image 3 (Columns A-L) - first 5 rows
    base_data_part1 = {
        'Ambient_Air_Temp': [100, 100, 100, 100, 100],
        'Ambient_Press': [19.8, 19.8, 19.8, 19.8, 19.9],
        'Charge_Pressure_mptr': [160, 162, 164, 166, 158],
        'ECM_Run_Speed': [3876423, 3876425, 3876427, 3876428, 3876430],
        'Engine_Speed_Flow': [1627.5, 1637.5, 1664, 1660.5, 1664.5],
        'Exhaust_Flow': [111.67, 114.03, 117.92, 118.61, 114.17],
        'Pedal_pos': [23.199, 20.797, 18.398, 18.797, 12],
        'Engine_mode': [11, 11, 11, 11, 11],
        'Key_Off_Count': [172, 172, 172, 172, 172],
        'Eng_torque': [464, 438, 455, 430, 335],
        'P_SFP_tr': [70.93, 70.93, 70.93, 70.93, 70.93],
        'Press_ur_Pump': [0.7, 2.2, 2.7, 21.6, 9.3]
    }

    # Base data from Image 2 (Columns M-Y) - first 5 rows
    base_data_part2 = {
        'Temp_OC_in': [269.9, 271.8, 273.4, 273.8, 272.1],
        'Temp_OC_out': [265.1, 264.7, 264.5, 264.7, 265],
        'Temp_DPF_out': [252.8, 253.6, 254.1, 254.6, 255.4],
        'Temp_DP_V_AIM_tr_C_urea_tank': [21.3, 21.3, 21.3, 21.3, 21.3],
        'V_ATM_f_g_HC_fb': [0, 0, 0, 0, 0],
        'V_ATM_f_V_DPF_bk_Total': [0.1638, 0.1651, 0.1692, 0.1707, 0.1704],
        'nox_in_ppm': [283, 281, 289, 283, 278],
        'NOx_out_ppm': [14, 14, 14, 15, 17],
        'Press_DP': [3.61, 3.73, 3.89, 3.99, 3.84],
        'Temp_SC': [226.1, 226.8, 227.7, 228, 228.4],
        'V_SCR_A': [233.1, 233.6, 234.3, 234.6, 235.3],
        'TR_DEF_inj_Rate': [0.055556, 0.05428, 0.0563, 0.09188, 0.05667]
    }

    # Base data from Image 1 (Columns Z-AJ) - first 5 rows
    base_data_part3 = {
        'Temp_SC_R_bed': [229.6, 230.2, 231, 231.3, 231.9],
        'V_SCR_A_NR_Fdbk': [0.696, 0.691, 1.312, 0.661, 0.681],
        'V_SCR_Ct_NH3_Slip_Detected': [0, 0, 0, 0, 0],
        'V_SFP_Ct_L_Soot': [96.85, 96.54, 96.25, 96.12, 95.78],
        'V_SFP_L_Soot_Load': [1.92, 1.92, 1.93, 1.93, 1.93],
        'GP_V_USM_dPM': [0.05412, 0.05433, 0.05693, 0.05647, 0.05467],
        'F_V_USM_gdpM': [0.06505, 0.06505, 0.13009, 0.06505, 0.06505],
        'Veh_speed_gfdbK': [69.19, 70.73, 70.44, 70.31, 70.89],
        'NOX_CE': [95.053, 95.01779, 95.15571, 94.69965, 93.84058]
    }

    df1_base = pd.DataFrame(base_data_part1)
    df2_base = pd.DataFrame(base_data_part2)
    df3_base = pd.DataFrame(base_data_part3)

    # Remove duplicate columns from subsequent dataframes before concatenation
    df2_base = df2_base.drop(columns=[], errors='ignore') # No explicit duplicates here
    df3_base = df3_base.drop(columns=[], errors='ignore') # No explicit duplicates here

    # Concatenate horizontally for the base 5 rows
    base_df = pd.concat([df1_base, df2_base, df3_base], axis=1)

    # Initialize an empty list to store generated rows
    generated_rows = []

    # Generate more rows by slightly varying the base data
    for _ in range(num_rows):
        row = base_df.sample(n=1).iloc[0].copy() # Randomly pick a base row

        # Apply small random noise to numerical columns to create variations
        for col in row.index:
            if pd.api.types.is_numeric_dtype(row[col]):
                # Add a small percentage of noise
                noise = row[col] * np.random.uniform(-0.01, 0.01)
                row[col] += noise
                
                # Ensure certain columns remain integers or within plausible ranges if necessary
                if col in ['Charge_Pressure_mptr', 'Key_Off_Count', 'Engine_mode']:
                    row[col] = int(round(row[col]))
                
                # Example: Ensure NOX_CE stays within 0-100
                if col == 'NOX_CE':
                    row[col] = np.clip(row[col], 0, 100)

        generated_rows.append(row)

    full_df = pd.DataFrame(generated_rows)
    
    # Ensure all columns have numerical types if possible
    for col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    return full_df

if __name__ == '__main__':
    df = get_sample_data()
    df.to_excel('data/sample_data.xlsx', index=False)
    print(f"Sample data successfully written to data/sample_data.xlsx with {len(df)} rows.")
