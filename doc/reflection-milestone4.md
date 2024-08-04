## Spotify Dashboard - Reflection

### Usability

- Based on the feedback received, the dashboard is very easy to use and navigate from tab to tab
- Some users overlooked some features and requested more direction, guided steps were added in the headers to clarify

### Reoccurring Themes

- Large dataset and multiple filters resulted in slower load times, however, preprocessed/partitioned data was utilized
- Keeping the format consistent with the headers, Spotify colours, and spacing/layout to be visually aesthetic

### Valuable Insights

- Include a link beside each top song so that they could sample the specific song - implemented in tab 2
- Include pictures of the top 3 artist in the world displayed depending what was selected - implemented in tab 3



### Tab 1 – Discover Music Taste

#### **What has been done:**

##### Improvements from Milestone 2
- Remove data/make the dataset callable when needed
- Make search area smaller/equal to the chart area beside
- Use Card element header instead of paragraph

##### Improvements from Milestone 3
- PREPROCESSED DATA to make navigation transitions and loading more efficient when filtering genres
- MODIFIED LAYOUT/STRUCTURE to handle the overflow string text of filters
- REFORMATTED colours, positioning/layout/sizing of the chart/table areas



### Tab 2 – Find New Music

#### **What has been done:**

##### Improvements from Milestone 2
- Rename ambiguous table columns, add mark ticks/slider position 
- Rearrange plot positions for maximum impact 
- Change background colors/Card colors to match the Spotify green theme

##### Improvements from Milestone 3
- ADDED scrollable table to optimize available space and allow the user to view more songs based on their selection
- ADDED hover "info" text to include a definition of the music metric term
- REFORMATTED colours, headers with steps to guide the user, positioning/layout/sizing of the chart/table areas


#### **What needs to be improved:**
- Add a checkbox to implement selecting rows based on popularity
    - this was not implemented due redundancy and lack of necessity



### Tab 3 – Explore the World

#### **What has been done:**

##### Improvements from Milestone 2
- Change background colors/Card colors to match the Spotify green theme
- Add second index to slider so a range of popularity can be selected
- Limit table columns to song, artist, and popularity

##### Improvements from Milestone 3
- CHANGED bar charts that display the top 10 songs and artist to SINGLE PIE CHART AND PICTURES OF TOP 3 ARTISTS
- MODIFIED main popularity slider to include a RANGE WITH 2 SELECTORS
    - Included for both global top 10 artists PIE CHART as well as the TOP 3 ARTIST PICTURES
    - Also controls the output for the top 10 songs when specific country selected
- REFORMATTED colours, headers with steps to guide the user, positioning/layout/sizing of the chart/table areas


#### **What needs to be improved:**

- Add an additional plot or summary statistics table in the bottom right
    - this was not implemented due to positioning, space, and layout of tab 3


