# <center>NBA GAME ATTENDANCE ANALYSIS</center>

### Usage 
in terminal in correct dir: conda env create -f environment.yml 

### Goal

- Analyze time-series data of game attendance and predict future attendances. Can base more complicated models off of additional features

### Data

- Scaped data from basketball-reference for game attendance for every game since 1991
  - this also included the two teams, the points scored by each team, the time, and which games were playoff games 
- Scaped data from wikipedia to get information on stadium capacities 
- Used information from wikipedia to create data for rivalries in the NBA
- Obtained Google Trend data for each team for the past 12 months [located in the data folder]

### Data Processing

- Created datetime type for game time
- Added col for if a game is a playoff game
- Removed all rows where a note existed 
  - Notes seemed only for when a game was held at an alternate venue as opposed to a team's stadium
- Typed data into correct types for values contained
- Added col for popularity, whether the home game was a win, day of the week, if match-up is a rivarly, attendance last game, and attendance two games ago
- Created a dictionary of dataframes for each team and only dates after the opening of their stadium
- Plotted Time Series for each team
  - need to analyze plots to decide if/which teams should be dropped
- Plotted hists of attendance for each team 
- Plotted bar charts of different variables for each team
- Plotted bar charts of attendance for days of the week
- Plotted autocorrelation for each team

### Modelling

- Numerous modelling methods can be used for each team's ts. 
  - A higher-order auto-regressive model
  - ARIMA Model
  - Seasonal Model
  - Random Forest
  - Others
  
### Some thoughts

- Could be used to estimate ticket price changes, or necessity of offering promos, giveaways, and other tactics to drive attendance

### Notes

- Many ideas and some guidance receieved from this article: https://www.researchgate.net/publication/320202978_Predicting_National_Basketball_Association_Game_Attendance_Using_Random_Forests

