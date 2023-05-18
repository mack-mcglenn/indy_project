## Rage in the (Nicolas) Cage
### An Exploration of the Nicolas Cage Movie Database

Nicolas Cage has undoubtedly one of the most eclectic filmographies in Hollywood. The same guy that falls in IMDB's top 75 percentile of highest paid actors also has an average Rotten Tomato score of less than 50%. And with over 100 screened roles credited to his name, it can certainly be said that Cage is a guy who is willing to try anything at least 4 times. This project will be mostly exploratory, taking a deep dive into the career of Ole Saint Nick. Then, I'll build a reccomendation system that will deliver suggestions of other films you might enjoy based off of similarity in Genre.
___________________________________________________________________________________________________________________________
### Deliverables

    Github Repo w/ Final Notebook and README
    Wrangle.py module to store the functions used in this project
    
### Project Outline

    Acquisition
    Preparation
    Exploration
    Analysis
    Suggestion Algorithm
    Conclusion

### Acquisition

Acquired from: Kaggle
Original Creator:  Em Highland (ehartlett)
Original Shape: 100 rows x 6 columns

![89B0D7EB-B782-43C6-84DB-5E0EDBAB33AF](https://github.com/mack-mcglenn/indy_project/assets/122935207/b0fa578c-4f44-43fc-9118-6ba50b8f59a7)

### Preparation

My first step in preparation was manually converting the original dataset into a csv. Then, I manually added the box office earnings for each film. My preparation also included the painstaking task of manually assigning a genre to each film as a feature. Then, I enumerated Genre and Rating. Within RottenTomatoes, my missing values are identified as 'X' rather than nulls, so I also dropped those values and converted RottenTomatoes to a float.

### Exploration

After I applied my preparation, I downloaded my new and improved dataframe into a .csv file called 'nic_cage_with_bo.csv'. That file included the box office (bo) earnings as well as the genres for each film. Then, I conducted my exploration in Tableau. After completing my exploration, I learned some interesting things. For example, Nicolas Cage has starred in nearly 2 dozen dramas, but his top earning genre is family films. Interasante!

### Analysis

Next step: analysis. I focused on two questions during this step:
1. Which genre brings an average of higher ratings for Nicky, thrillers or dramas?
2. Do Cage's later films get better or worse ratings than the earlier films?

### Suggestion Algorithm

The algorithm I built for this project runs as an input function. A user will input a film that they've seen, and the algorithm will output suggestions for two other films they might enjoy. Right now, the algorithm is based on similarity in genre. 

### Conclusion

This was a fun dataset to work with. I'd like to revisit it in the future to see how new films impact the overall results of my statistical analysis. I don't think that this is a dataset where modeling would be effective, but maybe I'm not asking the right questions yet. In addition, I'd like to revisit my suggestion function and see if I can build a dropdown menu in leui of the input function. I would also like to finetune the algorithm to base output on both genre similarity and RT score similarity. Til then, thank you for checking out my project! 

Here's a treat for making it all the way to the bottom:

![image](https://github.com/mack-mcglenn/indy_project/assets/122935207/960dad28-35cb-4efa-b249-2eac4b730f95)

