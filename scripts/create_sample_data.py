#!/usr/bin/env python3
"""
Generate sample data for testing the Danfoss RAG system.

Creates:
- Sample Excel file with part cross-references
- Sample CSV file with specifications
- Sample text file simulating a product guide
"""

import os
import random
from pathlib import Path

import pandas as pd


def create_sample_parts_crossref(output_dir: Path):
    """Create sample parts cross-reference Excel file."""

    # Sample data
    data = {
        'danfoss_part': [],
        'competitor_brand': [],
        'competitor_part': [],
        'description': [],
        'voltage': [],
        'current': [],
        'power_kw': [],
        'dimensions_mm': [],
        'category': []
    }

    # Danfoss part prefixes
    danfoss_prefixes = ['FC-051', 'FC-102', 'FC-302', 'MCD-201', 'MCD-500', 'VLT-6000']

    # Competitor brands and their part patterns
    competitors = {
        'Siemens': ['6SL3210', '6SE6440', 'MICROMASTER'],
        'ABB': ['ACS580', 'ACS880', 'ACS355'],
        'Schneider': ['ATV320', 'ATV930', 'ATV71'],
        'Yaskawa': ['GA500', 'GA700', 'A1000'],
        'WEG': ['CFW11', 'CFW500', 'CFW700']
    }

    categories = ['VFD', 'Soft Starter', 'Servo Drive', 'DC Drive']
    voltages = ['230V', '400V', '480V', '690V']
    powers = [0.75, 1.5, 2.2, 4.0, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]

    # Generate 50 sample parts
    for i in range(50):
        prefix = random.choice(danfoss_prefixes)
        power = random.choice(powers)

        # Create Danfoss part number
        danfoss_part = f"{prefix}P{power:.0f}K{'T4' if random.random() > 0.5 else 'T2'}E20H{'1' if random.random() > 0.5 else '2'}"

        # Create competitor part
        competitor_brand = random.choice(list(competitors.keys()))
        comp_prefix = random.choice(competitors[competitor_brand])
        competitor_part = f"{comp_prefix}-{random.randint(100, 999)}-{random.choice(['01', '02', '03'])}"

        # Voltage and current based on power
        voltage = random.choice(voltages)
        if voltage == '230V':
            current = round(power * 1000 / (230 * 0.9 * 1.732), 1)
        elif voltage == '400V':
            current = round(power * 1000 / (400 * 0.9 * 1.732), 1)
        else:
            current = round(power * 1000 / (480 * 0.9 * 1.732), 1)

        # Dimensions based on power
        if power <= 4:
            dims = f"{random.randint(100, 150)}x{random.randint(60, 100)}x{random.randint(120, 180)}"
        elif power <= 15:
            dims = f"{random.randint(150, 250)}x{random.randint(100, 180)}x{random.randint(200, 300)}"
        else:
            dims = f"{random.randint(250, 400)}x{random.randint(180, 280)}x{random.randint(300, 450)}"

        data['danfoss_part'].append(danfoss_part)
        data['competitor_brand'].append(competitor_brand)
        data['competitor_part'].append(competitor_part)
        data['description'].append(f"VFD {power}kW {voltage} Variable Frequency Drive")
        data['voltage'].append(voltage)
        data['current'].append(f"{current}A")
        data['power_kw'].append(power)
        data['dimensions_mm'].append(dims)
        data['category'].append(random.choice(categories))

    df = pd.DataFrame(data)

    output_path = output_dir / 'sample_parts_crossref.xlsx'
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Created: {output_path}")
    print(f"  - {len(df)} part cross-references")

    return output_path


def create_sample_specs_csv(output_dir: Path):
    """Create sample product specifications CSV."""

    data = {
        'product_code': [],
        'product_name': [],
        'input_voltage_min': [],
        'input_voltage_max': [],
        'output_frequency_min': [],
        'output_frequency_max': [],
        'ambient_temp_min': [],
        'ambient_temp_max': [],
        'ip_rating': [],
        'weight_kg': [],
        'efficiency_percent': []
    }

    products = [
        ('VLT Micro Drive FC 051', 'FC-051'),
        ('VLT HVAC Drive FC 102', 'FC-102'),
        ('VLT AutomationDrive FC 302', 'FC-302'),
        ('VLT Soft Starter MCD 201', 'MCD-201'),
        ('VLT Soft Starter MCD 500', 'MCD-500'),
    ]

    for name, code in products:
        for power in [1.5, 4.0, 7.5, 15, 30]:
            product_code = f"{code}P{power:.0f}K0T4E20"

            data['product_code'].append(product_code)
            data['product_name'].append(f"{name} {power}kW")
            data['input_voltage_min'].append(380)
            data['input_voltage_max'].append(480)
            data['output_frequency_min'].append(0)
            data['output_frequency_max'].append(590 if 'FC' in code else 60)
            data['ambient_temp_min'].append(-10)
            data['ambient_temp_max'].append(50 if power < 10 else 45)
            data['ip_rating'].append('IP20' if random.random() > 0.3 else 'IP54')
            data['weight_kg'].append(round(power * 0.8 + random.uniform(1, 5), 1))
            data['efficiency_percent'].append(round(97 + random.uniform(-1, 1), 1))

    df = pd.DataFrame(data)

    output_path = output_dir / 'sample_product_specs.csv'
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    print(f"  - {len(df)} product specifications")

    return output_path


def create_sample_user_guide(output_dir: Path):
    """Create a sample user guide text file (simulating PDF content)."""

    content = """
DANFOSS VLT MICRO DRIVE FC 051
QUICK START GUIDE

================================================================================
1. SAFETY INSTRUCTIONS
================================================================================

Before installing and operating the VLT Micro Drive, read this guide carefully.
All work on the drive must be performed by qualified personnel.

WARNING: High voltage! Risk of electric shock.
Wait at least 4 minutes after disconnecting power before servicing.

================================================================================
2. MECHANICAL INSTALLATION
================================================================================

2.1 Mounting Position
- The drive must be mounted vertically on a flat surface
- Ensure adequate cooling clearance (100mm top and bottom)
- IP20 units require installation in an IP54 or better enclosure

2.2 Dimensions
- FC-051P1K5: 100 x 68 x 155 mm (HxWxD)
- FC-051P4K0: 148 x 90 x 180 mm (HxWxD)
- FC-051P7K5: 195 x 110 x 220 mm (HxWxD)

2.3 Weight
- 1.5kW model: 1.2 kg
- 4.0kW model: 2.5 kg
- 7.5kW model: 4.8 kg

================================================================================
3. ELECTRICAL INSTALLATION
================================================================================

3.1 Mains Connection
- Input voltage: 200-240V 1-phase or 380-480V 3-phase
- Use appropriate cable sizes per local regulations
- Install line fuses or MCB as per specifications

3.2 Motor Connection
- U, V, W terminals for 3-phase motor connection
- PE terminal for protective earth
- Maximum cable length: 50m unscreened, 25m screened

3.3 Control Terminals
- 18: Digital input 1 (Start)
- 19: Digital input 2 (Reversing)
- 27: Digital input 3 (Coast Stop)
- 53: Analog input (0-10V / 4-20mA)
- 42: +24V DC output
- 39: Analog output

================================================================================
4. PROGRAMMING
================================================================================

4.1 Basic Parameters
- Parameter 1-20: Motor nominal power (kW)
- Parameter 1-22: Motor nominal voltage (V)
- Parameter 1-23: Motor nominal frequency (Hz)
- Parameter 1-24: Motor nominal current (A)
- Parameter 1-25: Motor nominal speed (RPM)

4.2 Quick Setup
1. Set motor nameplate data (par. 1-20 to 1-25)
2. Set application type (par. 1-00)
3. Perform AMA (par. 1-29)
4. Set reference source (par. 3-00)
5. Start and adjust as needed

================================================================================
5. TROUBLESHOOTING
================================================================================

Common Alarms and Warnings:

Alarm 2: No mains
- Check mains connection
- Check fuses

Alarm 7: DC overvoltage
- Increase deceleration time
- Enable DC brake function

Alarm 14: Earth fault
- Check motor cables
- Check motor insulation

Warning 8: DC undervoltage
- Check mains supply
- Check for voltage dips

================================================================================
6. SPECIFICATIONS
================================================================================

Environmental:
- Operating temperature: -10 to +50°C (derating above 45°C)
- Storage temperature: -25 to +65°C
- Humidity: 5-95% RH (non-condensing)
- Max altitude: 1000m (derating above)

Electrical:
- Input frequency: 50/60 Hz
- Output frequency: 0.2-590 Hz
- Switching frequency: 2-16 kHz
- Efficiency: >97%

Protection:
- Motor thermal protection via ETR
- Short circuit protection
- Earth fault protection
- Overload protection

================================================================================
For more information, visit www.danfoss.com/drives

Document number: MG02A402
Version: 1.0
Date: 2024-01
================================================================================
"""

    output_path = output_dir / 'sample_user_guide.txt'
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Created: {output_path}")
    print(f"  - User guide content ({len(content)} characters)")

    return output_path


def main():
    """Generate all sample data files."""

    # Create data directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  DANFOSS RAG - Sample Data Generator")
    print("=" * 60 + "\n")

    # Generate files
    create_sample_parts_crossref(data_dir)
    create_sample_specs_csv(data_dir)
    create_sample_user_guide(data_dir)

    print("\n" + "=" * 60)
    print(f"  All sample files created in: {data_dir}")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("  1. Start the backend: uvicorn backend.app.main:app --reload")
    print("  2. Ingest the sample data:")
    print(f"     python scripts/ingest_documents.py --dir {data_dir}")
    print("  3. Open frontend/demo.html to test the chatbot")


if __name__ == "__main__":
    main()
