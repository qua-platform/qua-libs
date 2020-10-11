import asyncio
import time
import random
from colorama import Fore


class Node:

    def __init__(self, label, prog=None):
        self.id = id(self)
        self.label = label
        self.rest = {'rest': random.random()}
        self.prog = prog

    async def run(self):
        print(random.choice(colors) + f"Running node {self.label} for {self.rest['rest']} seconds")
        output = await self.prog(**self.rest)
        print(f"DONE with node {self.label}")
        # return output


class Graph:

    def __init__(self, nodes, edges, start):
        self.id = id(self)
        self.nodes = nodes
        self.edges = edges
        self.start = start.id

    def run(self):
        for node in self.nodes.values():
            node.run()

    def get_next(self):
        to_do = [self.start]
        while to_do:
            s = self.nodes[to_do.pop(0)]
            yield s
            try:
                for child in self.edges[s.id]:
                    to_do.append(child)
            except KeyError:
                pass


async def prog(rest):
    await asyncio.sleep(rest)


colors = list(vars(Fore).values())
a = Node('a', prog)
b = Node('b', prog)
c = Node('c', prog)
d = Node('d', prog)
nodes = {node.id: node for node in [a, b, c, d]}
edges = {a.id: [b.id, c.id], b.id: [d.id], c.id: [d.id]}
graph = Graph(nodes, edges, a)


async def main(graph):
    a = [asyncio.create_task(n.run()) for n in graph.get_next()]
    await asyncio.gather(*a)
    return a

s = time.perf_counter()
f = asyncio.run(main(graph))
print(Fore.WHITE + f"Total took: {time.perf_counter() - s}")
