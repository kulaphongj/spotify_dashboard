## Spotify Dashboard - Reflection

### Tab 1 – Discover Music Taste

**What has been done:**

- Genres, artists, track names, filters
- Statistics table of music, including min, mean, and max, are shown in the table based on the filtering
- Genre proportion based on the filtering
- Radar chart of scaled values based on the filtering
- If the filters are not selected, the values of the table and charts will be calculated based on all data

**Limitations:**

- Slow loading (track names list of the filter contains about 80,000)

**What needs to be improved:**
- Remove data/make the dataset callable when needed (stored variable)
- Make search area smaller/equal to the chart area beside
- Use Card element header instead of paragraph

### Tab 2 – Find New One?

**What has been done:**

- Created sliders for each music feature on the left pane which interacts with the plots on the right pane (Feature Bar Plot, Feature Bar Plot, and the Song Table)
- When the sliders are used to select feature ranges, Feature Bar Plot will be updated with the average musical feature proportions, showing the trends
- When the sliders are used to select feature ranges, Feature Bar Plot will be updated to reflect the musical features showing fine-tuned visualizations
- When the sliders are used to select feature ranges, Song Table gets updated with the top 10 songs based on the feature ranges selected

**What needs to be improved:**

- Rename ambiguous table columns
- Add mark ticks/slider position
- Potential rearranging of the plot positions for maximum impact
- Change background colors/Card colors to match the Spotify green theme
- Add a checkbox to implement selecting rows based on popularity

### Tab 3 – Explore Globally

**What has been done:**

- Created interactive map where users can select specific countries to display the top 10 songs based on popularity (e.g., top 10 songs rated 100, 50 etc.)
- Created 2 horizontal bar charts that display the top 10 songs and artist globally
- Created main popularity slider to control the output that is presented
    - Included for both global top 10 songs and top 10 artists horizontal bar charts
    - Also controls the output for the top 10 songs when specific country selected
- Incorporated additional package (pycountry) that converts ISO-alpha3 country codes (required for map display) into full country names for interpretability

**What needs to be improved:**

- Change background colors/Card colors to match the Spotify green theme
- Add second index to slider (so there are 2) so a range of popularity can be selected
- Limit table to popularity or include 3/4 metrics (mainly covered in first 2 tabs)
    - If so, potentially add an additional plot or summary statistics in the bottom right to further describe popularity insights

### General Concerns

- Data loading issues
- Flexible sizing or fixed (if can make scrollable, fixed might be better)
- When table exceeds background area, how to make background conform new width
