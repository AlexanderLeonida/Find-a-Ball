with open("circle_data.txt", "w") as file:
    x_coordinate, y_coordinate = 350, 350
    for i in range(100):
        for i in range(50): 
            file.write(f"({x_coordinate}, {y_coordinate}):")
            x_coordinate += 1
            y_coordinate += 1
        for i in range(50): 
            file.write(f"({x_coordinate}, {y_coordinate}):")
            x_coordinate -= 1
            y_coordinate += 1
        for i in range(50): 
            file.write(f"({x_coordinate}, {y_coordinate}):")
            x_coordinate -= 1
            y_coordinate -= 1
        for i in range(50): 
            file.write(f"({x_coordinate}, {y_coordinate}):")
            x_coordinate += 1
            y_coordinate -= 1
        