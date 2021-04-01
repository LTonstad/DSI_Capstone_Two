
  * Can get game/season stats using [Sports Reference API](https://sportsreference.readthedocs.io/en/stable/) that pulls from [Sportsreference.com](www.sports-reference.com)
    * [Feature Descriptions](https://sportsipy.readthedocs.io/en/latest/nba.html#module-sportsipy.nba.player)

    > * Most categories that end with `_percentage` (besides `Field-Goals`) are referring to `Stat per 100 possessions`
    > * Features include: `assist_percentage`, `block_percentage`, `defensive_rebound_percentage`, `offensive_rebound_percentage`, `steal_percentage`, `total_rebound_percentage`, `turnover_percentage`, `usage_percentage`
    > * `Salary` gives career earnings


* **Goal(s)**: Cluster NBA players into different positional categories based off season stats, then train Neural Network to predict something like win percentage based off salary distribution within a team (or perhaps have it categorize into something like [Great Team, Good Team, Okay Team, Bad Team, Awful Team] if that is more feasible)
 * *Bonus*: Compare results to the current ongoing season

## Exploratory Data Analysis

Used functions to create DataFrames from this [Towards DataScience Article](https://towardsdatascience.com/sports-reference-api-intro-dbce09e89e52)

* These functions created two DataFrames:
  * Season DataFrame:

    > * 94 Columns (`29 null columns` that were removed)
    > * 13,235 Rows
    > * 2,033 Unique Players

  * Career DataFrame:
  
    > * 94 Columns (`29 null columns` that were removed)
    > * 2,040 Rows
    > * 2,033 Unique Players (Down to 1923 players after removing players without salaries)

  * Null Columns:
  
  > ['and_ones', 'blocking_fouls', 'dunks', 'lost_ball_turnovers',
'net_plus_minus', 'offensive_fouls', 'on_court_plus_minus',
'other_turnovers', 'passing_turnovers', 'percentage_field_goals_as_dunks',
'percentage_of_three_pointers_from_corner', 'percentage_shots_three_pointers',
'percentage_shots_two_pointers', 'points_generated_by_assists',
'shooting_fouls', 'shooting_fouls_drawn', 'shots_blocked',
'take_fouls', 'team_abbreviation', 'three_point_shot_percentage_from_corner',
'three_pointers_assisted_percentage', 'two_pointers_assisted_percentage',
'age', 'player_id', 'center_percentage', 'point_guard_percentage',
'power_forward_percentage', 'shooting_guard_percentage',
'small_forward_percentage', 'year']

There does appear to be 353 players that have no positions assigned, could try to have it predict the correct positions based on that data?

## Notes

* Would like to add the following columns: ~~current_year_salary, current_player(boolean), current_team~~

* Try to incorporate [TSNE Plot](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

* Some examples found [here](https://thevi5ion.wordpress.com/2017/07/13/classifying-nba-players-using-machine-learning-in-python/)

* Possible new goal: Train model to predict a players salary this current season based on their current seasons stats, have someone in audience choose player from this season & have model predict then check to verify in their actual contract

* Using all values was generally a bad idea, was suggesting between 2-3 clusters

* Players that are not currently playing do not have contract data, but can find contracts using [spotrac](https://www.spotrac.com/nba/)