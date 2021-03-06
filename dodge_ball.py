import math
import random
import numpy as np
from collections import defaultdict
import uuid
import mesa
import numpy as np
import pandas as pd
from mesa import space
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement, UserSettableParameter
from mesa.visualization.modules import ChartModule

import matplotlib.pyplot as plt

class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
    ] 

    def __init__(self, canvas_height=500,
                 canvas_width=500, instantiate=True):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.identifier = "space-canvas"
        if (instantiate):
            new_element = ("new Simple_Continuous_Module({}, {},'{}')".
                           format(self.canvas_width, self.canvas_height, self.identifier))
            self.js_code = "elements.push(" + new_element + ");"

    def portrayal_method(self, obj):
        return obj.portrayal_method()

    def render(self, model):
        representation = defaultdict(list)
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.pos[0] - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.pos[1] - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min)) 
            representation[portrayal["Layer"]].append(portrayal)
        return representation

def wander(x, y, r, speed, model,team):
    """
    Let a agent wander around the environment

    Args:
        x (int): x coordinate of the agent
        y (int): y coordinate of the agent
        r (int): 
        speed (int): speed of the agent
        model (model): model of the environment
        team (bool): team of the agent

    Returns:
        array: new position of the agent 
    """
    
    margin=20
    
    r = r + (-2)*(random.random()<0.5)*r+(random.random()-0.5)*math.pi/2
    new_x = max(min(x + math.cos(r) * speed-300*team, model.space.x_max-300-margin)+300*team, model.space.x_min+margin+300*team)
    new_y = max(min(y + math.sin(r) * speed, model.space.y_max-margin), model.space.y_min+margin)
    
    return np.array([new_x, new_y]),r

class  Game(mesa.Model):
    def  __init__(self,  n_player, param1=None, param2=None):
        mesa.Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)

        team1, team2 = create_team(n_player)
        #Uncomment the next line to create a team with the best skills possible
        # team1 = create_best_team(n_player)

        #Uncomment the next line to create a team with preselected parameters 1 and 2
        # team1, team2 = create_team_benchmark(n_player, param1, param2,'strength','speed')

        team1[:,0] = team1[:,0] * 20 + 10
        team2[:,0] = team2[:,0] * 20 + 10
        team1[:,1] = team1[:,1] * 15 + 40
        team2[:,1] = team2[:,1] * 15 + 40
        team1[:,3] = np.minimum(team1[:,3], 1)
        team2[:,3] = np.minimum(team2[:,3], 1)

        self.team1 = team1
        self.team2 = team2
        print("team : player :   |speed  |  strength  |  precision  |  caught  | " )
        for  i  in  range(n_player):
            speed, strength, precision,caught = self.team1[i]
            self.schedule.add(Player(x=random.random() * 300 + 300,  y=random.random()  *  600,  speed=speed, team=True,  strength=strength, precision=precision, caught=caught, unique_id=uuid.uuid1(), model=self))
            print("team 1 player : ", i + 1," {:.2f}".format(speed),"     {:.2f}".format(strength),"       {:.2f}".format(precision),"         {:.2f}".format(caught))
        for  i  in  range(n_player):
            speed, strength, precision,caught = self.team2[i]

            self.schedule.add(Player(x=random.random() * 300,  y=random.random()  *  600,  speed=speed, team=False,  strength=strength, precision=precision, caught=caught, unique_id=uuid.uuid1(), model=self))
            print("team 2 player : ", i + 1," {:.2f}".format(speed),"     {:.2f}".format(strength),"       {:.2f}".format(precision),"         {:.2f}".format(caught))
            
        
        self.schedule.add(Ball(random.random()  *  300,  random.random()  *  300,  5 , 0, random.random()<0.5 ,uuid.uuid1(), self))    
         
        
        self.datacollector = DataCollector(model_reporters={
            "team1": [lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and not player.team and not player.touched]),[self]],
            "team2": [lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and player.team and not player.touched]),[self]]})
        



    def step(self):	
        if (not sum([1 for player in self.schedule.agent_buffer() if player.is_player and player.team]) or not 	
         sum([1 for player in self.schedule.agent_buffer() if player.is_player and not player.team])): 	
            self.running = False	
            return	
        self.schedule.step()	
        self.datacollector.collect(self)	
        if self.schedule.steps >= 1000:	
            self.running = False


def create_best_team(n_player):
    """
    Create the best team for the game

    Args:s
        n_player (int): number of players in the team

    Returns:
        array: return a team with the best parameters
    """
    attribute = ['speed', 'strength', 'precision','caught']
    n = len(attribute)
    team = []
    for j in range(n_player):
            skill_list = []
            for k in range(n):
                skill =  1.5 #In a gaussian distribution, it corresponds to the top 1% of the distribution
                skill_list.append(skill)
            team.append(skill_list)

    team_array = np.array(team)
    return team_array

def create_team(n_player):
    """
    Create a team of n_player players with fair distribution of the team using softmax function
    Args:
        n_player (int): number of players in the team
        Returns:

        array: array of players with their skills
    """
    n_team = 2
    attribute = ['speed', 'strength', 'precision','caught']
    team_1 = []
    team_2 = []
    n_player = 6
    n = len(attribute)
    for i in range(n_team):
        for j in range(n_player):
            skill_list = []
            for k in range(n):
                skill =  np.random.normal()
                skill_list.append(skill)
            if i==0:
                team_1.append(skill_list)
            else:
                team_2.append(skill_list)

    team_1_array = np.array(team_1)
    team_2_array = np.array(team_2)
    return np.exp(team_1_array +np.random.random(4)) / np.sum(np.exp(team_1_array), axis=0), np.exp(team_2_array+np.random.random(4)) / np.sum(np.exp(team_2_array), axis=0)

def create_team_benchmark(n_player, param1, param2, skill_type1, skill_type2):
    """
    create a team of n_player players while setting parameters 1 and 2 and choosing the rest randomly

    Args:
        n_player (int): number of players in the team
        param1 (int): value of the first parameter 
        param2 (int): value of the second parameter 
        skill_type1 (str): name of the first parameter 
        skill_type2 (str): name of the second parameter

    Returns:
        tuple: return two teams  
    """
    n_team = 2
    attribute = ['speed', 'strength', 'precision','caught']
    team_1 = []
    team_2 = []
    n_player = 6
    n = len(attribute)
    idx1 = attribute.index(skill_type1)
    idx2 = attribute.index(skill_type2)
    for i in range(n_team):
        for j in range(n_player):
            skill_list = []
            for k in range(n):
                skill =  np.random.normal()
                skill_list.append(skill)
            if i==0:
                skill_list[idx1] = param1
                team_1.append(skill_list)
            else:
                skill_list[idx2] = param2
                team_2.append(skill_list)

    team_1_array = np.array(team_1)
    team_2_array = np.array(team_2)
    return np.exp(team_1_array +np.random.random(4)) / np.sum(np.exp(team_1_array), axis=0), np.exp(team_2_array+np.random.random(4)) / np.sum(np.exp(team_2_array), axis=0)


def dist_seg(P,A,B,direction):
    """
    Compute the distance between a point P and a segment AB

    Args:
        P (array): point
        A (array): point
        B (array): point
        direction (int): angle

    Returns:
        float: distance between the point and the segment
    """
    BH=np.dot(A-B,direction)*direction
    PB=B-P
    PA=A-P
    PH=PB+BH

    return min(min(np.linalg.norm(PA),np.linalg.norm(PB)),np.linalg.norm(PH))


class Player(mesa.Agent):
    def __init__(self, x, y, speed, team,strength, precision, caught, unique_id: int, model: Game):
        """
        Class of the players of dodgeball

        Args:
            x (int): x coordinate of the player
            y (int): y coordinate of the player
            speed (float): speed of the player
            team (bool): team of the player
            strength (float): strength of the player 
            precision (float): precision of the player
            caught (float): probability to catch a ball
            unique_id (int): unique id of the player 
            model (Game): model of the game
        """
        super().__init__(unique_id, model)
        self.pos = np.array([x, y])
        self.initial_speed = speed
        self.speed = speed
        self.model = model
        self.team = team
        
        self.facing = math.pi/2
        self.is_player = True

        self.strength = strength
        self.precision = precision
        self.touched = False
        self.caught = caught 
        self.has_ball = False
        self.can_get_ball = False
        self.ball = None

        self.size = 7.5
        
    def portrayal_method(self):
        
        color = "blue"
        r = self.size
        
        if self.team: 
            
            color = "red" 
        

        portrayal = {"Shape": "circle",
                     "Layer": 1, 
                     "Color": color,
                     "r": r}
        return portrayal

    def step(self):
        
        if self.can_get_ball:
            # if the player can get the ball, he will get it
            u=self.ball.pos-self.pos
            d=np.linalg.norm(u)
            if d<self.speed:
                self.pos=self.ball.pos
                self.has_ball = True
                self.can_get_ball = False
                
            else: 

                self.pos=self.pos+(u/d)*self.speed


        elif self.has_ball : 
            # if the player has the ball, he will throw the ball
            r=random.random()*2*math.pi
            precision_error=(1-self.precision)*17.5*np.array([np.cos(r),np.sin(r)])

            u=random.choice([player for player in self.model.schedule.agent_buffer() if player.team!=self.team and player.is_player]).pos-self.pos+precision_error
            d=np.linalg.norm(u)

            self.can_get_ball = False
            self.ball.direction=u/d
            self.ball.speed=self.strength 
            self.ball.on_ground = False
            self.has_ball=False
            self.ball.thrower_team=self.team
            self.ball.is_getting_picked = False
            self.ball.thrower=self

        else: 
            # if the player doesn't have the ball, he will move
            self.pos,self.facing= wander(self.pos[0], self.pos[1], self.facing ,self.speed, self.model, self.team)
            [setattr(self,"pos",self.pos+(2.1)*self.size*(self.pos - player.pos)/np.linalg.norm(self.pos - player.pos)) for player in self.model.schedule.agent_buffer() if np.linalg.norm(self.pos - player.pos)  < 16 and player.is_player and player!=self ]


        self.speed = self.initial_speed*math.exp(-self.model.schedule.steps/400)
        
class Ball(mesa.Agent):
    def __init__(self, x, y, width, speed, team ,unique_id, model):
        """
        Class of the ball of dodgeball

        Args:
            x (int): x coordinate of the ball
            y (int): y coordinate of the ball
            width (int): width of the ball
            speed (float): speed of the ball
            team (bool): team of the ball
            unique_id (int): unique id of the ball
            model (class): model of the game
        """
        super().__init__(unique_id, model)
        self.pos = np.array([x+300*team,y])
        self.previous_pos = np.array([x+300*team,y])
        self.model = model
        self.width = width
        self.is_player = False
        self.on_ground = True
        self.team = team
        self.speed = speed
        self.direction = None
        self.thrower_team = None
        self.is_getting_picked = False
    
    def portrayal_method(self):
        color = "black"
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": self.width}
        return portrayal


    def step(self):
        
        
        if self.speed<5:
            # if the ball is stopped, it is set as on the ground and it can be picked
            # by player in the corresponding playground
            self.on_ground = True
            self.speed = 0
            self.is_getting_picked=False
            if self.pos[0]<300:
                
                self.team = False
            else:
                self.team = True
        
        else :
            # if the ball is moving, it is moving in the direction of the speed
            self.previous_pos = self.pos
            self.pos=self.pos+self.speed*self.direction
            self.speed = self.speed * 0.9


        if (self.pos<0).any() or (self.pos>600).any():
            # if the ball is out of the playground, it reappears in the playground
            self.team = not self.team
            new_thrower=random.choice([player for player in self.model.schedule.agent_buffer() if player.team==self.team and player.is_player])
            setattr(new_thrower,"ball",self)
            setattr(new_thrower,"has_ball",True)
            self.pos=new_thrower.pos
            self.thrower_team = self.team 
            self.speed = 0

        if self.on_ground and not self.is_getting_picked:
            # if the ball is on the ground, it can be picked by the closest player 
            player_distance=[[np.linalg.norm(player.pos-self.pos),player] for player in self.model.schedule.agent_buffer() if player.team==self.team and player.is_player]
            closest_player=min(player_distance,key=lambda x : x[0])[1]
            setattr(closest_player,"can_get_ball",True)
            setattr(closest_player,"ball_pos",self.pos)
            setattr(closest_player,"ball",self)
            self.is_getting_picked=True
        if not self.on_ground:

            if np.array([dist_seg(player.pos,self.previous_pos,self.pos,self.direction)<player.size+self.width for player in self.model.schedule.agent_buffer() if player.is_player and player!=self.thrower]).any():
            
                
                self.on_ground=True
                self.is_getting_picked=False
                player_hit=[player for player in self.model.schedule.agent_buffer() if player.is_player and dist_seg(player.pos,self.previous_pos,self.pos,self.direction)<player.size+self.width][0]
                if player_hit.team!=self.thrower_team:

                    p=player_hit.strength/self.speed*player_hit.caught*(0.5+0.5*((12.5-dist_seg(player_hit.pos,self.previous_pos,self.pos,self.direction))/12.5))
                   
                    if random.random()>p:
                        self.model.schedule.remove(player_hit)
                    else:
                        self.model.schedule.remove(self.thrower)
                self.speed=0

def run_single_server():
    """
    Function to run the model in a single server
    """
    
    chart = ChartModule([{"Label": "team1", "Color": "blue"},
                         {"Label": "team2", "Color": "red"},],
                         data_collector_name= 'datacollector',canvas_height=200,canvas_width=500)

    s_player = UserSettableParameter("slider","nb_of_players", 6, 0, 10, 1)
    


    server  =  ModularServer(Game, [ContinuousCanvas(),chart],"Game",{"n_player": s_player})
    server.port = 8521 
    server.launch()  

def plot(title,xlabel,ylabel,winner, loser):
    """
    Function to plot the results of the model

    Args:
        title (str): title of the plot
        xlabel (str): label of the x axis
        ylabel (str): label of the y axis
        winner (list): stats about the team that won the game
        loser (list): stats about the team that lost the game
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.scatter(range(len(winner)), winner , label="Winner")
    plt.scatter(range(len(loser)), loser, label="Loser")
    plt.legend()
    plt.show()


def run_batch(benchmark=False):
    """
    Function to run the model in a batch, to compare the different parameters


    Args:
        benchmark (bool): if True, the model is run in benchmark mode

    Returns:
        dataframe: dataframe containing the results of the model
    """
    if benchmark:
        param1 = np.linspace(0,1,10)
        param2 = np.linspace(0,1,10)
    else:
        param1 = None
        param2 = None

    n_simulation = 100
    batchrunner = BatchRunner(Game, {'n_player': n_simulation * [6],  'param1': param1,'param1': param2},
                       model_reporters={"nb_of_players_team_1": lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and not player.team and not player.touched ]),
                                        "nb_of_players_team_2": lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and player.team and not player.touched] ), 
                                        "winner": lambda x: 1 if sum([1 for player in x.schedule.agent_buffer() if player.is_player and player.team and not player.touched]) > sum([1 for player in x.schedule.agent_buffer() if player.is_player and not player.team and not player.touched]) else 2 if  sum([1 for player in x.schedule.agent_buffer() if player.is_player and player.team and not player.touched]) < sum([1 for player in x.schedule.agent_buffer() if player.is_player and not player.team and not player.touched]) else 3,                                        
                                        "max_caught_team_1": lambda x: np.max(x.team1[:,3]),
                                        "max_caught_team_2": lambda x: np.max(x.team2[:,3]),
                                        "min_caught_team_1": lambda x: np.min(x.team1[:,3]),
                                        "min_caught_team_2": lambda x: np.min(x.team2[:,3]),
                                        "mean_caught_team_1": lambda x: np.mean(x.team1[:,3]),
                                        "mean_caught_team_2": lambda x: np.mean(x.team2[:,3]),
                                        "mean_speed_team_1": lambda x: np.mean(x.team1[:,0]),
                                        "mean_speed_team_2": lambda x: np.mean(x.team2[:,0]),
                                        "mean_strength_team_1": lambda x: np.mean(x.team1[:,1]),
                                        "mean_strength_team_2": lambda x: np.mean(x.team2[:,1]),
                                        "mean_precision_team_1": lambda x: np.mean(x.team1[:,2]),
                                        "mean_precision_team_2": lambda x: np.mean(x.team2[:,2]),
                                        "max_speed_team_1": lambda x: np.max(x.team1[:,0]),
                                        "max_speed_team_2": lambda x: np.max(x.team2[:,0]),
                                        "max_strength_team_1": lambda x: np.max(x.team1[:,1]),
                                        "max_strength_team_2": lambda x: np.max(x.team2[:,1]),
                                        "max_precision_team_1": lambda x: np.max(x.team1[:,2]),
                                        "max_precision_team_2": lambda x: np.max(x.team2[:,2]),
                                        "min_speed_team_1": lambda x: np.min(x.team1[:,0]),
                                        "min_speed_team_2": lambda x: np.min(x.team2[:,0]),
                                        "min_strength_team_1": lambda x: np.min(x.team1[:,1]),
                                        "min_strength_team_2": lambda x: np.min(x.team2[:,1]),
                                        "min_precision_team_1": lambda x: np.min(x.team1[:,2]),
                                        "min_precision_team_2": lambda x: np.min(x.team2[:,2])


                                        
                                        })


    batchrunner.run_all()
    df = batchrunner.get_model_vars_dataframe()

    print(df)
    

    winner_caught = []
    loser_caught = []
    winner_speed = []
    loser_speed = []
    winner_strength = []
    loser_strength = []
    winner_precision = []
    loser_precision = []


    draw_stats = []
    for index, row in df.iterrows():
        if row['winner'] == 1:
            winner_caught.append(row['mean_caught_team_1'])
            loser_caught.append(row['mean_caught_team_2'])

            winner_speed.append(row['mean_speed_team_1'])
            loser_speed.append(row['mean_speed_team_2'])

            winner_strength.append(row['mean_strength_team_1'])
            loser_strength.append(row['mean_strength_team_2'])

            winner_precision.append(row['mean_precision_team_1'])
            loser_precision.append(row['mean_precision_team_2'])

            

        elif row['winner'] == 2:
            winner_caught.append(row['mean_caught_team_2'])
            loser_caught.append(row['mean_caught_team_1'])

            winner_speed.append(row['mean_speed_team_2'])
            loser_speed.append(row['mean_speed_team_1'])

            winner_strength.append(row['mean_strength_team_2'])
            loser_strength.append(row['mean_strength_team_1'])

            winner_precision.append(row['mean_precision_team_2'])
            loser_precision.append(row['mean_precision_team_1'])



    plot("Caught", "Number of simulation", "Mean caught", winner_caught, loser_caught)
    plot("Speed", "Number of simulation", "Mean speed", winner_speed, loser_speed)
    plot("Strength", "Number of simulation", "Mean strength", winner_strength, loser_strength)
    plot("Precision", "Number of simulation", "Mean precision", winner_precision, loser_precision)

    return df
    


if  __name__  ==  "__main__":
    
    #server.port = 8521
    #server.launch()
    
    # To run the mesa interface
    run_single_server()

    # Uncomment the next line to run the script in batch and to save the results of the simulation in a scrpt
    # df = run_batch()
    # df.to_csv("data.csv")
