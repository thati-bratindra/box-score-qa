# Extending the pipeline with player tracking data

The next data source I would add is player tracking data: 29 skeletal points per player at 60 frames per second for a full game. A coach could ask, "Is our center slower getting back on defense in the fourth quarter?" This program only knows whole-game totals, so plan.py can’t answer quarter questions. Tracking data can show how open a shooter was, which lineups spread the floor the best, or early signs of injury.

I'd store raw camera data as one file per game, then convert it into events like "Player 12 sprinted back at 10 mph with 3:41 left in the fourth." The AI model still fills in a form of operations over an events table. Every event would be validated against trustworthy stats: if tracking says 30 minutes played, the box score should too.

This would currently break because embed.py and rag.py find answers by meaning. embed.py turns each row into a sentence, like "Tavian Belcourt scored 55 points," and saves what it means. rag.py then finds the rows that sound most like the question. That works because a box-score row is one complete fact. One camera frame is a bunch of dots. A sprint exists across hundreds of frames, so the answer has to be looked up and not calculated.
