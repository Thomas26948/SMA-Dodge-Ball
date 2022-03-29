from enum import unique
import math
import random
from re import L
from tkinter import W
import numpy as np
from collections import defaultdict

import uuid
import mesa
import numpy
import pandas
import matplotlib.pyplot as plt
from mesa import space
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import ChartModule

from mesa.visualization.ModularVisualization import UserSettableParameter

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
    
    

def wander(x, y, speed, model,team):
    '''
    Randomly move in one of the cardinal directions.

    Args:
        x: Starting x coordinate
        y: Starting y coordinate
        speed: How fast to move
        model: The model in which the agent lives
        team: the team of the agent

    Returns:
        A coordinate that is one step closer to the destination.
    '''
    r = random.random() * math.pi * 2
    new_x = max(min(x + math.cos(r) * speed, model.space.x_max), model.space.x_min)
    new_y = max(min(y + math.sin(r) * speed, model.space.y_max), model.space.y_min)

    if team == 1: # team 1 should stay in the top of the map
        while (new_y > model.h/2):
            r = random.random() * math.pi * 2
            new_x = max(min(x + math.cos(r) * speed, model.space.x_max), model.space.x_min)
            new_y = max(min(y + math.sin(r) * speed, model.space.y_max), model.space.y_min)
    else:# team 2 should stay in the bottom of the map
        while (new_y < model.h/2):
            r = random.random() * math.pi * 2
            new_x = max(min(x + math.cos(r) * speed, model.space.x_max), model.space.x_min)
            new_y = max(min(y + math.sin(r) * speed, model.space.y_max), model.space.y_min)

    return new_x, new_y

def count_human(model):
    '''
    count the number of human in the model
    '''
    cpt = 0
    for agent in model.schedule.agent_buffer():
        if agent.lp == False:
            cpt += 1
    return cpt





class  Playgound(mesa.Model):
    def  __init__(self,  n_player_team_1,n_player_team_2,n_ball):
        """
        Initialize a new model with the given parameters.
        Args:
            n_player_team_1 (int): number of player in team 1
            n_player_team_2 (int): number of player in team 2
            n_ball (int): number of ball
        """
        mesa.Model.__init__(self)
        self.h = 600
        self.w = 600
        self.space = mesa.space.ContinuousSpace(self.h, self.w, False)
        self.schedule = RandomActivation(self)


        for  _  in  range(n_player_team_1):
            self.schedule.add(Player(uuid.uuid1(),self, random.random()  *  self.h,  random.random()  *  int(self.w/2),  10,50 ,40,5,5,False,100,1,team=1))
        for  _  in  range(n_player_team_2):
            self.schedule.add(Player(uuid.uuid1(),self, random.random()  * self.h,  random.random()  *  int(self.w/2) + int(self.w/2),  10,50 ,40,5,5,False,100,1,team=2))
        for  _  in  range(n_ball):
            self.schedule.add(Ball(uuid.uuid1(),self, random.random()  *  self.h,  random.random()  *  self.w,  10))

    def step(self):
        # self.dc.collect(self)
        self.schedule.step()


        
        if self.schedule.steps >= 1000:
            self.running = False



class Player(mesa.Agent):
    def __init__(self, unique_id, model, x, y, speed, power, distance_attack=40, height=5, precision=5, touched=False, endurance=100, p_good_receip=1, team=1, has_ball=False):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.power = power
        self.speed = speed
        self.model = model
        self.distance_attack = distance_attack
        self.height = height
        self.precision = precision
        self.touched = touched
        self.endurance = endurance
        self.p_good_receip = p_good_receip
        self.team=team
        self.has_ball = has_ball
        self.is_player = True
    
    def portrayal_method(self):
        if self.team == 1:
            color = "red"
        else:
            color = "blue"

        if self.touched == True:
            color = "orange"
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": self.height}
        return portrayal
    
    def search_ball(self):
        ball_accessible = [ ball for ball in self.model.schedule.agent_buffer() if ( ball.is_player == False) and  (( (ball.pos[0] - self.pos[0] ) **2) + ( (ball.pos[1] - self.pos[1]) ** 2 ) ) < 100**2 ]
        return ball_accessible


    def step(self) -> None:
        if self.touched: # if touched by the ball remove the agent from the model
            self.model.schedule.remove(self)
            self.model.space.remove_agent(self)
            return
        else:
            if not self.has_ball:
                self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model,self.team)
                ball_accessible = self.search_ball()
                if ball_accessible:
                    self.has_ball = True





class Ball(mesa.Agent):
    def __init__(self, unique_id, model, x, y,width):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.model = model
        self.width = width
        self.is_player = False
        self.is_taken = False

    
    def portrayal_method(self):
        color = "black"
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": self.width}
        return portrayal


# class Villager(mesa.Agent):
#     def __init__(self, x, y, speed, unique_id: int, model: Village, distance_attack=40, p_attack=0.6,loup_garou=False):
#         super().__init__(unique_id, model)
#         self.pos = (x, y)
#         self.speed = speed
#         self.model = model
#         self.distance_attack = distance_attack
#         self.p_attack = p_attack
#         self.lp = loup_garou
#         self.r = 3
#         self.lyco_transformed = False

    

#     def step(self):
#         self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)
#         self.transform()
        
#     def transform(self):
#         if self.lp == True:
#             if random.random() < 0.1 and self.lyco_transformed == False:
#                 self.r = 6
#                 self.lyco_transformed = True
#             if self.lyco_transformed == True and random.random() < self.p_attack:
#                 self.attack()
                
    
#     def attack(self):
#         villager_attackable = [ villager for villager in self.model.schedule.agent_buffer() if ( villager.lp == False) and  (( (villager.pos[0] - self.pos[0] ) **2) + ( (villager.pos[1] - self.pos[1]) ** 2 ) ) < self.distance_attack**2 ]
#         # We attack someone randomly from the list:
#         # print(len(villager_attackable))
#         if len(villager_attackable) != 0:
#             vil = np.random.choice(villager_attackable)
#             vil.lp = True


# class Cleric(Villager):
#     def __init__(self, x, y, speed, unique_id: int, model: Village, distance_savable=30, p_save=0.6,loup_garou=False):
#         super(Villager,self).__init__(unique_id, model)
#         Villager.__init__(self,x,y,speed,unique_id,model)      
#         self.distance_savable = distance_savable
#         self.p_save = p_save

#     def portrayal_method(self):
#         if self.lp == True:
#             color = "red"
#         else:
#             color = "green"
        
#         portrayal = {"Shape": "circle",
#                      "Filled": "true",
#                      "Layer": 1,
#                      "Color": color,
#                      "r": self.r}
#         return portrayal

#     def step(self):
#         self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)
#         self.transform()

#         if random.random() < self.p_save and self.lp == False: 
#             villager_savable = [ villager for villager in self.model.schedule.agent_buffer() if villager.lyco_transformed == False and (( (villager.pos[0] - self.pos[0] ) **2) + ( (villager.pos[1] - self.pos[1]) ** 2 ) ) < (self.distance_savable)**2 ]
#             #We save someone randomly from the list:
#             if len(villager_savable ) != 0:
#                 vil = np.random.choice(villager_savable)
#                 vil.lp = False


# class Hunter(Villager):
#     def __init__(self, x, y, speed, unique_id: int, model: Village, distance_huntable=40, p_hunt=0.6,loup_garou=False):
#         super(Villager,self).__init__(unique_id, model)
#         Villager.__init__(self,x,y,speed,unique_id,model) 
#         self.distance_huntable = distance_huntable
#         self.p_hunt = p_hunt

#     def portrayal_method(self):
#         if self.lp == True:
#             color = "red"
#         else:
#             color = "black"
        
#         portrayal = {"Shape": "circle",
#                      "Filled": "true",
#                      "Layer": 1,
#                      "Color": color,
#                      "r": self.r}
#         return portrayal

#     def step(self):
#         self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)

#         self.transform()


#         if random.random() < self.p_hunt and self.lp == False: 

#             lyco_huntable = [ lyco for lyco in self.model.schedule.agent_buffer() if lyco.lyco_transformed == True and (( (lyco.pos[0] - self.pos[0] ) **2) + ( (lyco.pos[1] - self.pos[1]) ** 2 ) ) < (self.distance_huntable)**2 ]
#             # We hunt a lyco randomly from the list:
#             if len(lyco_huntable) != 0:
#                 lyco = np.random.choice(lyco_huntable)
#                 self.model.schedule.remove(lyco)




def run_single_server():
    
    # chart = ChartModule([{"Label": "human_count", "Color": "Blue"},{"Label": "lycanthrope_count", "Color": "Red"},
    # {"Label": "transformed_lycanthrope_count", "Color": "Purple"},{"Label": "agent_count", "Color": "Yellow"}],
    # data_collector_name= 'dc',canvas_height=200,canvas_width=500)

    n_player_team_1_slider = UserSettableParameter('slider',"Number of player in the first team", 5, 0, 20, 1)
    n_player_team_2_slider = UserSettableParameter('slider',"Number of player in the second team", 5, 0, 20, 1)
    n_ball_slider = UserSettableParameter('slider',"Number of ball", 1, 0, 10, 1)

    # server  =  ModularServer(Playgound, [ContinuousCanvas(),chart],"Playground",{"n_player_team_1":  n_player_team_1_slider,"n_player_team_2": n_player_team_2_slider,

    server  =  ModularServer(Playgound, [ContinuousCanvas()],"Playground",{"n_player_team_1":  n_player_team_1_slider,"n_player_team_2": n_player_team_2_slider,
     "n_ball": n_ball_slider})
    server.port = 8521
    server.launch()

# def run_batch():
#     params_dict = {
#     'n_villagers': [50],
#     'n_lp' : [5],
#     'n_hunter' : [1],
#     'n_cleric' : range(0,6,1) }

#     br = BatchRunner(Village,params_dict,
#     model_reporters=
#     {"human_count": lambda m : count_human(m) , "lycanthrope_count": lambda m : count_lycanthrope(m), 
#     "transformed_lycanthrope_count": lambda m : count__transformed_lycanthrope(m),
#     "agent_count": lambda m : m.schedule.get_agent_count()   })
    
#     br.run_all()
#     df = br.get_model_vars_dataframe()
#     print(df)
#     df[['human_count','lycanthrope_count','transformed_lycanthrope_count','agent_count']].plot()
#     plt.show()



if  __name__  ==  "__main__":
    # server  =  ModularServer(Village, [ContinuousCanvas(),chart],"Village",{"n_villagers":  n_villagers_slider,"n_lp":n_lycanthropes_slider,
    #  "n_cleric":n_clerics_slider,"n_hunter":n_hunters_slider})
    # server.port = 8521
    # server.launch()

    # run_batch()

    run_single_server()
