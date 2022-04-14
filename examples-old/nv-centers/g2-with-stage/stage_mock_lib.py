class stage_mock:
    def __init__(self):
        self.name = "stage1"
        self.x = 0
        self.y = 0

    def __repr__(self):
        return f"{self.name} at {self.x}/{self.y}"

    def go_to(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        print(f"set pos to {self.x}/{self.y}")

    def get_pos(self):
        return self.x, self.y
