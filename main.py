import pygame
import pymysql
from dotenv import load_dotenv
import os
import atexit
import time
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from lstm_model import generate_multistep_data, train_lstm_model
from linear_regression_models import train_position_model, train_velocity_model


load_dotenv()
db = pymysql.connect(host='localhost', user='root', password=os.getenv('pwrd'), database='HTMDNA')
db.autocommit(True)
crsr = db.cursor()

CREATE_TABLE_SQL = """
    CREATE TABLE `Coordinates` (
        `x_coordinate` float,
        `y_coordinate` float
    );
"""
DROP_TABLE_SQL = """
    DROP TABLE Coordinates;
"""
INSERT_COORDINATES_SQL = """
    INSERT INTO `Coordinates` (x_coordinate, y_coordinate) VALUES (%s, %s);
"""

#  https://www.geeksforgeeks.org/how-to-create-buttons-in-pygame/
def start_button(screen):
    font = pygame.font.Font(None, 50)
    # white text
    text = font.render("Start", True, (255, 255, 255))
    # dimensions
    button_rect = pygame.Rect(540, 310, 200, 100)
    # red button
    pygame.draw.rect(screen, (255, 0, 0), button_rect)
    # render text onto screen
    # messed around to find these coordinates, they are in the middle of the screen shifted to the left
    screen.blit(text, (600, 340))
    pygame.display.flip()
    return button_rect

def wasd_instruction_button(screen):
    font = pygame.font.Font(None, 50)
    # white text 
    instruction_text = font.render("Use WASD to move for 5 seconds", True, (255, 255, 255))
    # dimensions
    instruction_rect = pygame.Rect(300, 450, 680, 100)
    # semi-transparent black background for instruction box
    pygame.draw.rect(screen, (0, 0, 0, 128), instruction_rect)
    # render instruction text onto screen
    # these are shifted even more to the left
    screen.blit(instruction_text, (350, 475))
    pygame.display.flip()

def wait_for_start(screen):
    wasd_instruction_button(screen)
    button_rect = start_button(screen)
    waiting = True
    # same check as in both draw methods
    # wait until button is clicked
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False

def ask_model_choice(screen):
    font = pygame.font.Font(None, 40)
    screen.fill("black")

    text1 = font.render("Press '0' for Linear Regression", True, (255, 255, 255))
    text2 = font.render("Press '1' for LSTM Model", True, (255, 255, 255))
    screen.blit(text1, (400, 300))
    screen.blit(text2, (400, 350))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    return "linear"
                elif event.key == pygame.K_1:
                    return "lstm"


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Track a Ball")
    clock = pygame.time.Clock()
    return screen, clock

def create_coordinates_table():
    crsr.execute(CREATE_TABLE_SQL)

def drop_coordinates_table():
    crsr.execute(DROP_TABLE_SQL)

def save_coordinates(xcoord, ycoord):
    crsr.execute(INSERT_COORDINATES_SQL, (xcoord, ycoord))

def on_exit():
    drop_coordinates_table()
    crsr.close()
    db.close()
    pygame.quit()
    print("tables deleted")

def update_player_position(player_pos, dt):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player_pos.y -= 500 * dt
    if keys[pygame.K_s]:
        player_pos.y += 500 * dt
    if keys[pygame.K_a]:
        player_pos.x -= 500 * dt
    if keys[pygame.K_d]:
        player_pos.x += 500 * dt

    # boundaries for the ball so that it cannot go off of the sight of the screen, streamlined
    player_pos.x = max(50, min(1230, player_pos.x))
    player_pos.y = max(50, min(670, player_pos.y))

# draw circle & background for user to move ball around on 
def userDraw(screen, player_pos):
    screen.fill("black")
    pygame.draw.circle(screen, "white", player_pos, 50)
    pygame.draw.circle(screen, "red", player_pos, 5)

# draw circle & background for AI to move ball around on 
def computerDraw(screen, ai_pos):
    screen.fill("black")
    pygame.draw.circle(screen, "red", ai_pos, 50)
    pygame.draw.circle(screen, "black", ai_pos, 5)

def userInputDrawing(player_pos, screen, clock, user_input_time_limit):
    # to be used to stop the running program after time limit is up
    running = True
    dt = 0
    start_time = time.time()
 
    player_trajectory = []
 
    while running:
        elapsed_time = time.time() - start_time
        # check elapsed time to automatically kill user input
        if elapsed_time > user_input_time_limit:
            running = False
       
        # from pygame vector in main
        xcoord = int(player_pos[0] // 1)
        ycoord = int(player_pos[1] // 1)
 
        player_trajectory.append((xcoord, ycoord))
        save_coordinates(xcoord, ycoord)
 
        # kill game on esc button or window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
 
        # move player via framerate (dt), visualize it on screen with draw method
        update_player_position(player_pos, dt)
        userDraw(screen, player_pos)
 
        pygame.display.flip()
 
        dt = clock.tick(60) / 1000
 
 
    return player_trajectory


def AIOutputDrawing(ai_pos, screen, clock, output_time_limit, position_model, velocity_model_x, velocity_model_y, player_trajectory):
    running = True
    dt = 0
    # track time steps for AI, movement over time
    start_time = time.time()
    time_step = 0
 
    # checks if there are previous positions recorded, if there are, set ai_pos to there. Will start drawing AI movement based on last recorded position from user input
    # if there are no coordinates, set ai_pos to middle of screen (either the ball hasn't moved, or something has gone horribly wrong)
    ai_pos.x, ai_pos.y = player_trajectory[-1] if player_trajectory else (screen.get_width() / 2, screen.get_height() / 2)
 
    # to slow the AI prediction down, logic bug here, it just accelerates slower to make it look like AI is moving correctly
    # in reality, the AI is over estimating trajectories causing a fast increase of movement slope
    speed_multiplier = 0.5 
 
 
    while running: 
        # check if AI drawing time is up 
        elapsed_time = time.time() - start_time
        if elapsed_time > output_time_limit:
            running = False
 
        # new inpuy of time dimension per every loop cycle to be used for velocity prediction
        input_data = np.array([[ai_pos.x, ai_pos.y, time_step]])
 
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        # same citation as method train_position_model
        # predicted_position = position_model.predict(input_data)

        # to be used if we wanted to teleport the ball to the position
        # predicted_x, predicted_y = predicted_position[0]
 

        # sets new velocities based on previous velocities
        # because of this, over time, the AI will move faster, assuming that it's own velocity is increasing each time
        velocity_x = velocity_model_x.predict(input_data).item() * speed_multiplier 
        velocity_y = velocity_model_y.predict(input_data).item() * speed_multiplier 
 
        # move position of ball based on velocities so it doesn't teleport
        ai_pos.x += velocity_x
        ai_pos.y += velocity_y
 
        # make sure AI doesn't go out of bounds
        ai_pos.x = max(50, min(1230, ai_pos.x))
        ai_pos.y = max(50, min(670, ai_pos.y))
 
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
 

        computerDraw(screen, ai_pos)
 
        pygame.display.flip()
        dt = clock.tick(60) / 1000 
        time_step += 1 

def write_coordinates_to_file():
    # Open (or create) a file in write mode ('w')
    with open("./training_data/coordinates.txt", "w") as file:
        # Read data from the database
        df = pd.read_sql_query("SELECT * FROM Coordinates", db)
        #print(df)

        # format each row as (x, y)
        coordinates = [f"({x}, {y})" for x, y in zip(df['x_coordinate'], df['y_coordinate'])]

        # add the delimeter for txt file parsing
        formatted_coordinates = ":".join(coordinates)

        file.write(formatted_coordinates)

        # return df to use afterwards in main
        return df

def main():
    atexit.register(on_exit)
    screen, clock = init_pygame()
    create_coordinates_table()
    
    wait_for_start(screen)
    model_choice = ask_model_choice(screen)

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    ai_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    player_trajectory = userInputDrawing(player_pos=player_pos, screen=screen, clock=clock, user_input_time_limit=15)
    print("User input done")

    df = write_coordinates_to_file()

    if model_choice == "linear":
        position_model = train_position_model(df)
        velocity_model_x, velocity_model_y = train_velocity_model(df)

        if not position_model or not velocity_model_x:
            return

        AIOutputDrawing(
            ai_pos=ai_pos,
            screen=screen,
            clock=clock,
            output_time_limit=50,
            position_model=position_model,
            velocity_model_x=velocity_model_x,
            velocity_model_y=velocity_model_y,
            player_trajectory=player_trajectory
        )

    elif model_choice == "lstm":

        # build and train LSTM model
        lstm_model = train_lstm_model("coordinates.txt")
        print(lstm_model)
        print("______________________________________________________")

        predicted_trajectory = []
        for _ in range(len(lstm_model[0])):
            predicted_trajectory.append((int(lstm_model[0][_]) - 1, int(lstm_model[1][_]) - 1))

        # animate the predicted path
        print(predicted_trajectory)
        for _ in predicted_trajectory:
            ai_pos = pygame.Vector2(_[0], _[1])
            computerDraw(screen, ai_pos)
            pygame.display.flip()
            # frames per second, increase to make go faster
            clock.tick(20)

    sns.scatterplot(x='x_coordinate', y='y_coordinate', data=df)
    plt.gca().invert_yaxis()
    plt.title('Ball Trajectory')
    plt.show()

    print("Program finished.")

if __name__ == main():
    main()