#!/usr/bin/env python3
"""
Quick Start Example

This is a simple script to get you started with satellite data download.
It downloads a true color image of Lake Bled, Slovenia.

Before running:
1. Install dependencies: pip install sentinelhub matplotlib numpy python-dotenv
2. Create a .env file with your Sentinel Hub credentials:
   SH_CLIENT_ID=your_client_id_here
   SH_CLIENT_SECRET=your_client_secret_here
3. Get free credentials at: https://apps.sentinel-hub.com/

Author: Learning Script
Date: 2025
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
)

def main():
    """Quick start example for satellite data download"""
    print("=== Satellite Data Quick Start ===\n")
    
    # Load credentials from .env file
    load_dotenv()
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    
    # Check credentials
    if not config.sh_client_id or not config.sh_client_secret:
        print("❌ Error: Missing credentials!")
        print("Please create a .env file with:")
        print("SH_CLIENT_ID=your_client_id_here")
        print("SH_CLIENT_SECRET=your_client_secret_here")
        print("\nGet free credentials at: https://apps.sentinel-hub.com/")
        return
    
    print("✅ Credentials loaded successfully")
    
    # Define area of interest (Lake Bled, Slovenia)
    bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)
    print(f"📍 Area: Lake Bled, Slovenia {bbox}")
    
    # Define time range
    time_interval = ('2023-07-01', '2023-07-31')
    print(f"📅 Time: {time_interval[0]} to {time_interval[1]}")
    
    # Create evalscript for true color RGB
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    
    print("🛰️  Downloading Sentinel-2 image...")
    
    # Create and execute request
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=(512, 512),
        config=config,
    )
    
    try:
        # Download the image
        image = request.get_data()[0]
        
        print("✅ Download successful!")
        print(f"📊 Image shape: {image.shape}")
        print(f"📊 Data type: {image.dtype}")
        
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title('Sentinel-2 True Color Image\nLake Bled, Slovenia', fontsize=14)
        plt.axis('off')
        
        # Add some text information
        plt.figtext(0.02, 0.02, 
                   f'Data: Sentinel-2 L2A\nTime: {time_interval[0]} to {time_interval[1]}\nBands: B04 (Red), B03 (Green), B02 (Blue)', 
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.show()
        
        print("\n🎉 Success! You've downloaded your first satellite image!")
        print("\nNext steps:")
        print("- Try changing the bbox to your area of interest")
        print("- Experiment with different time periods")
        print("- Run other example scripts for advanced features")
        print("- Check the README.md for more detailed instructions")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("\nPossible solutions:")
        print("- Check your internet connection")
        print("- Verify your credentials are correct")
        print("- Try a different time period (some dates may not have data)")
        print("- Check if the area has data coverage")

if __name__ == "__main__":
    main()
