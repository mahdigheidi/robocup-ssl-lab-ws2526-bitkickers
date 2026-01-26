class Field:
    division_to_dims = {
        # 'DivisionName': [x_max, y_max, goal_width, penalty_width, penalty_depth, radius]
        'A': [6000, 4500, 1800, 3600, 1800, 500],
        'B': [4500, 3000, 1000, 2000, 1000, 500],
        'S': [1500, 1000, 800, 667, 334, 350],
        'S_Easy': [1500, 1000, 2000, 667, 334, 350]
    }

    def __init__(self, div_name):
        self.div_name = div_name
        field_data = Field.division_to_dims.get(div_name, 'B')
        self.x_max = field_data[0]
        self.x_min = -self.x_max
        self.y_max = field_data[1]
        self.y_min = -self.y_max
        self.goal_width = field_data[2]
        self.penalty_width = field_data[3]
        self.penalty_depth = field_data[4]
        self.radius = field_data[5]

    def __repr__(self):
        str_repr = f"Division {self.div_name}\n" \
            + f"Half-length: {self.x_max} Half-width: {self.y_max}\n" \
            + f"GoalWidth: {self.goal_width}, Penalty width, depth: {self.penalty_width}, {self.penalty_depth}"
        return str_repr


if __name__ == '__main__':
    s = Field('S')
    b = Field('B')
    print(s)
    print(b)
