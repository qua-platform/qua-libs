import uuid


class QuantumRouter:
    def __init__(self, id, top=0, routing=0, right=0, left=0):
        self.id = id
        self.top = top
        self.routing = routing
        self.right = right
        self.left = left

    def __repr__(self):
        return f"id={self.id},top={self.top},routing={self.routing},right=[{self.right}],left=[{self.left}]"

    def initialization(self, val):
        pass

    def downstream(self, val):
        pass

    def upstream(self, val):
        pass

    def extraction(self, val):
        pass

    def bottom_of_tree(self):
        if isinstance(self.right, QuantumRouter) & isinstance(self.left, QuantumRouter):
            return False
        if isinstance(self.right, QuantumRouter) ^ isinstance(self.left, QuantumRouter):
            raise Exception(AttributeError, "Badly formed quantum register!")
        else:
            return True


class QRAM:
    def __init__(self, address_width: int):
        self.address_width = address_width
        self.root = self.make_tree(address_width)

    def make_tree(self, depth):

        node = QuantumRouter(id=uuid.uuid4())
        if depth == 1:
            return node
        node.left = self.make_tree(depth - 1)
        node.right = self.make_tree(depth - 1)
        return node

    def __repr__(self):
        return f"{self.address_width} qubit QRAM"


if __name__ == '__main__':
    qram = QRAM(3)
    print(qram.root)