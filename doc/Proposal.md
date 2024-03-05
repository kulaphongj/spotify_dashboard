# "Discover Your Music Taste" Dashboard Proposal

## Section 1: Motivation and Purpose

**Motivation:** The world of music on Spotify offers endless opportunities for discovery, yet many users find themselves trapped within the confines of algorithm-driven recommendations. Our project seeks to democratize music discovery by providing tools that allow users to explore music based on its intrinsic qualities rather than solely on popularity metrics. By doing so, we aim to enrich the music listening experience, offering a deeper understanding of musical preferences and uncovering hidden gems in the Spotify catalog.

**Purpose:** The Spotify Tracks Genre Dashboard is designed to serve music enthusiasts who wish to delve deeper into their musical tastes and discover new music that resonates with them on a more personal level. Our focus is on visualizing the musical elements that draw users to certain songs and examining how the popularity of tracks compares to their musical features in influencing user preferences.

**Target Audience:** Our dashboard caters to two main groups:
- Curious Music Fans: Beginners eager to understand the "why" behind their favorite songs.
- Adventurous Listeners: Experienced users looking to broaden their musical horizons beyond mainstream recommendations.

**Role:** As data scientists, we aim to transform the Spotify dataset into a powerful tool for musical exploration, discovering the nuanced aspects of music that connect with listeners beyond the charts.

## Section 2: Description of the Data

**Dataset Overview:** Our project uses the [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset/data) from Kaggle, encompassing over 114000 tracks with rich metadata and audio features. This dataset provides a comprehensive foundation for analyzing musical trends, preferences, and the underlying features that define genres.

**Additional Dataset:** In addition to the Spotify Tracks Genre Dataset, we may also merge it with another dataset available on Kaggle ([Top Spotify Songs in 73 Countries](https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated)). This dataset includes additional columns such as 'daily_rank', 'daily_movement', 'weekly_movement', 'country', and 'snapshot_date'. These columns contribute to the popularity metrics and provide insights into the daily and weekly movements of songs in various countries.

**Key Features:**
- **Track and Genre Information:** Essential for categorizing music and facilitating genre-based exploration.
- **Audio Features:** Includes tempo, danceability, energy, and other metrics that quantify the characteristics of each track, crucial for visualizing the musical elements that appeal to users.
- **Popularity Metrics:** Offers insight into the commercial success of tracks, allowing us to explore the relationship between popularity and musical features.

**Project Utilization:** We will leverage these datasets to:
- Visualize the distribution of audio features within users' preferred music, identifying patterns and preferences.
- Analyze the impact of popularity on user preferences, distinguishing between tracks favored for their intrinsic qualities versus their commercial success.
- Incorporate additional data from the second dataset to enhance our analysis of popularity metrics across different countries and time periods.

## Section 3: Research Questions and Usage Scenarios

**Main Research Questions:**
1. What musical elements (tempo, energy, danceability, etc.) are most influential in shaping user preferences?
2. How does the popularity of tracks relate to their musical features in influencing user choices?

**Additional Research Questions:**
1. **Relationship between Popularity and Musical Features**: 
   - How do musical features such as danceability, energy, acousticness, and valence correlate with the popularity of tracks across different countries?
   - Are there specific combinations of musical features that contribute most to the popularity of tracks?
2. **Temporal Analysis of Popularity and Musical Features**: 
   - How do the popularity and musical features of tracks vary over time, as reflected in daily and weekly movements in different countries?
   - Are there specific time signatures or tempos that are more prevalent in popular tracks during different periods?
3. **Genre-specific Analysis**: 
   - How do the distributions of musical features differ across various genres of music?
   - Are there distinctive patterns in the musical features of tracks belonging to different genres, and how do these patterns influence popularity?
4. **Explicit Content and Popularity**: 
   - Does the presence of explicit content in tracks have an impact on their popularity across different countries?
   - Are there differences in the distribution of explicit tracks based on their popularity levels?
5. **Regional Preferences and Popularity**: 
   - How do regional preferences, as indicated by daily rank movements in different countries, affect the popularity of tracks with specific musical features?
   - Are there variations in the popularity of tracks based on their acousticness, instrumentalness, or tempo across different countries?
6. **Longitudinal Analysis of Popularity**: 
   - How do the popularity levels of tracks change over time, and are there specific musical features that are associated with long-term popularity trends?
   - Can changes in the popularity of tracks be predicted based on their musical features and historical performance?

**Usage Scenario:** Michael Jackson, a curious music fan, uses our dashboard to explore the audio features of his favorite tracks. He discovers that he has a preference for high-energy, danceable music, a pattern he hadn't recognized before. He then compares his favorites to popular tracks and realizes that his preferences align more with musical features than with popularity scores. Inspired by these insights, he explores recommendations generated by his unique "musical DNA," discovering new artists and genres that match his identified preferences.

**Secondary Goal:** If time permits, we will implement a feature allowing users like Michael Jackson to receive personalized music recommendations based on their preferred musical elements, enhancing the music discovery process.

## Additional Resources

- [Description of the App & Sketch](README.md)
- [Team Contract](team-contract.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [License](LICENSE)
