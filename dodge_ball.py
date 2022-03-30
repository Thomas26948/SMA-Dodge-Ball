import math
import random
import numpy as np
from collections import defaultdict

import uuid
import mesa
import numpy
import pandas
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
    
    margin=20
    
    r = r + (-2)*(random.random()<0.5)*r+(random.random()-0.5)*math.pi/2
    new_x = max(min(x + math.cos(r) * speed-300*team, model.space.x_max-300-margin)+300*team, model.space.x_min+margin+300*team)
    new_y = max(min(y + math.sin(r) * speed, model.space.y_max-margin), model.space.y_min+margin)
    
    return np.array([new_x, new_y]),r

class  Game(mesa.Model):
    def  __init__(self,  n_player, n_ball):
        mesa.Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)
        for  _  in  range(n_player):
            self.schedule.add(Player(random.random()  *  300 + 300,  random.random()  *  600,  25, True,  10, 0.75, 0.75, uuid.uuid1(), self))
        for  _  in  range(n_player):
            self.schedule.add(Player(random.random()  *  300,  random.random()  *  600,  25, False, 10, 0.75, 0.75, uuid.uuid1(), self))    
        for  _  in  range(n_ball):
            self.schedule.add(Ball(random.random()  *  300,  random.random()  *  300,  5 , 0, random.random()<0.5 ,uuid.uuid1(), self))    
         
        
        self.datacollector = DataCollector(model_reporters={
            "team1": [lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and not player.team]),[self]],
            "team2": [lambda x: sum([1 for player in x.schedule.agent_buffer() if player.is_player and player.team]),[self]]})
        
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.schedule.steps >= 1000:
            self.running = False

class Player(mesa.Agent):
    def __init__(self, x, y, speed, team,strength, precision, caught, unique_id: int, model: Game):
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
        self.ball_pos = None

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
            
            u=self.ball_pos-self.pos
            d=np.linalg.norm(u)
            if d<self.speed:
                self.pos=self.ball_pos
                self.has_ball
                self.can_get_ball = False

            else: 

                self.pos=self.pos+(u/d)*self.speed


        elif self.has_ball :

           pass

        else: 
            self.pos,self.facing= wander(self.pos[0], self.pos[1], self.facing ,self.speed, self.model, self.team)
            
            [setattr(self,"pos",self.pos+(2.1)*self.size*(self.pos - player.pos)/np.linalg.norm(self.pos - player.pos)) for player in self.model.schedule.agent_buffer() if np.linalg.norm(self.pos - player.pos)  < 16 and player != self ]
        

        self.speed = self.initial_speed*math.exp(-self.model.schedule.steps/200)
        
class Ball(mesa.Agent):
    def __init__(self, x, y, width, speed, team ,unique_id, model):
        super().__init__(unique_id, model)
        self.pos = np.array([x+300*team,y])
        self.model = model
        self.width = width
        self.is_player = False
        self.is_taken = False
        self.on_ground = True
        self.team = team
        self.speed = speed
        self.destination = None
        

    
    def portrayal_method(self):
        color = "black"
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": self.width}
        return portrayal


    def step(self):

        if self.on_ground:

            player_distance=[[np.linalg.norm(player.pos-self.pos),player] for player in self.model.schedule.agent_buffer() if player.team==self.team and player!=self]
            closest_player=min(player_distance,key=lambda x : x[0])[1]
            setattr(closest_player,"can_get_ball",True)
            setattr(closest_player,"ball_pos",self.pos)
            print(closest_player)
        
def run_single_server():
    
    chart = ChartModule([{"Label": "team1", "Color": "blue"},
                         {"Label": "team2", "Color": "red"},],
                         data_collector_name= 'datacollector',canvas_height=200,canvas_width=500)

    s_player = UserSettableParameter("slider","nb_of_players", 6, 0, 10, 1)
    s_ball = UserSettableParameter("slider","nb_of_balls", 1, 1, 10, 1)



    server  =  ModularServer(Game, [ContinuousCanvas(),chart],"Game",{"n_player": s_player, "n_ball": s_ball})
    server.port = 8521 
    server.launch()  


# def run_batch():
    
    
#     batchrunner = BatchRunner(Village, {'n_villagers': [50],"n_werewolves" : [5],"n_cleric" : list(range(0,6,1)),'n_hunter' : [1]},
#                        model_reporters={"nb_of_villagers": lambda x: sum([1 for villager in x.schedule.agent_buffer() if not villager.iswerewolf]),
#                                         "nb_of_werewolves": lambda x: sum([1 for villager in x.schedule.agent_buffer() if villager.iswerewolf and not villager.istransformed] ), 
#                                         "nb_of_transformed_werewolves": lambda x: sum([1 for villager in x.schedule.agent_buffer() if villager.iswerewolf and villager.istransformed] ), 
#                                         "nb_of_characters": lambda x: sum([1 for villager in x.schedule.agent_buffer()] )})
    
#     batchrunner.run_all()
#     df = batchrunner.get_model_vars_dataframe()
#     return df
    

if  __name__  ==  "__main__":
    #server  =  ModularServer(Village, [ContinuousCanvas()],"Village",{"n_villagers":  20,"n_werewolves":  5,"n_cleric":  1,"n_hunter":  2})
    #server.port = 8521
    #server.launch()
    
    
    run_single_server()
    #df=run_batch()
