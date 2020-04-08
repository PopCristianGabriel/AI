import os
from asyncio import sleep

import pygame
import neat
import random

pygame.init()
birdSize = 30
winWidth = 1280
winHeight = 720
generation = 0
pipeHeight = winHeight
pipeWidth = 100

birdsImg = [pygame.image.load("bird1.png"), pygame.image.load("bird2.png"), pygame.image.load("bird3.png")]
bgImg = pygame.transform.scale((pygame.image.load("bg.png")), (1280, 720))
pipeImg = pygame.transform.scale((pygame.image.load("pipe.png")), (100, winHeight))
pipeImgUpsideDown = pygame.transform.scale((pygame.image.load("pipe2.png")), (100, winHeight))
baseImg = pygame.image.load("base.png")


class Bird:
    images = birdsImg
    maxRotation = 25
    maxVel = -10
    animationTime = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rotation = 0
        self.tick_count = 0
        self.vel = 10
        self.firstAnimation = 10
        self.secondAnimation = 20
        self.thirdAnimation = 30
        self.currentAnimation = self.images[0]
        self.displacement = 0



    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -9.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        self.displacement = self.vel * (self.tick_count) + 0.4 * 3.3 * (self.tick_count) ** 2  # calculate displacement

        # terminal velocity
        if self.displacement >= 8:
            self.displacement = 8

        if self.displacement < 0:
            self.displacement -= -4

        self.y = self.y + self.displacement

    def draw(self, win):

        self.animationTime += 1

        if (self.animationTime <= self.firstAnimation):
            self.currentAnimation = self.images[0]
        elif (self.animationTime <= self.secondAnimation):
            self.currentAnimation = self.images[1]
        elif (self.animationTime <= self.thirdAnimation):
            self.currentAnimation = self.images[2]
        else:
            self.animationTime = 0

        if (self.y > winHeight - 30):
            self.y = winHeight - 30
        win.blit(self.currentAnimation, (self.x, self.y))
        pygame.display.update()

    def get_input(self):

        if (pygame.key.get_pressed()[pygame.K_SPACE]):
            if (self.displacement > 0):
                # print("jump " + str(self.displacement))
                self.jump()


class Pipe:
    gapHeight = 130

    def __init__(self):
        self.x = winWidth
        self.gap = self.choose_gap()
        self.image = pipeImg
        self.imageUpsideDown = pipeImgUpsideDown

    def choose_gap(self):
        q = random.randrange(-200, 400)
        print(q)
        return q

    def get_first_pipe(self):
        return -winHeight / 2 - self.gap

    def get_second_pipe(self):
        return self.gapHeight - self.gap

    def move(self):
        self.x -= 5

    def draw(self, win):
        win.blit(self.imageUpsideDown, (self.x, -winHeight / 2 - self.gap))
        win.blit(self.image, (self.x, winHeight / 2 + self.gapHeight - self.gap))

    def passed(self, bird):
        return self.x + pipeWidth <= 0


def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


def draw_window(win, birds, pipes, generation,genomes):
    win.blit(bgImg, (0, 0))
    font = pygame.font.SysFont("comicsansms", 72)
    text = font.render("generation : " + str(generation), True, (0, 128, 0))
    win.blit(text,
             (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    fitnesses = []
    for x,g in enumerate(genomes):
        fitnesses.append(genomes[x].fitness)
    maximum = max(fitnesses)
    for x,g in  enumerate(genomes):
        if(g.fitness == maximum):
            birds[x].draw(win)



    pygame.display.update()


def colide(pipe, bird):
    if bird.y >= winHeight - 10:
        print("1")
        print("1")
        print("1")
        return True

    if bird.x >= pipe.x and bird.x <= pipe.x + pipeWidth and bird.y < -winHeight / 2 - pipe.gap + pipeHeight:
        print("2")
        print("2")
        print("2")
        return True

    if bird.x >= pipe.x and bird.x <= pipe.x + pipeWidth and bird.y > winHeight / 2 + pipe.gapHeight - pipe.gap:
        print("3")
        print("3")
        print("3")
        return True

    if bird.x + birdSize >= pipe.x and bird.x <= pipe.x + pipeWidth and bird.y + birdSize > winHeight / 2 + pipe.gapHeight - pipe.gap:
        print("4")
        print("4")
        print("4")
        return True

    if (bird.y >= winHeight - 30):
        print("5")
        print("5")
        print("5")
        return True

    if (bird.y <= 0):
        return True

    return False



def main(genomes, config, g=None):
    ge = []
    nets = []
    global generation
    generation += 1
    birds = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    pipes = []
    pipes.append(Pipe())
    pipes.append(Pipe())
    pipes[1].x += 700
    win = pygame.display.set_mode((winWidth, winHeight))
    surface = pygame.display.get_surface()
    ww, wh = pygame.display.get_surface().get_size()

    clock = pygame.time.Clock()
    while (True):
        clock.tick(60)
        pygame.event.get()
        # print(bird.rotation)
        # bird.get_input()
        for pipe in pipes:
            pipe.move()

        pipe_ind = 0
        if (len(birds) > 0):
            if (len(pipes) > 1):
                if (birds[0].x > pipes[0].x + pipeWidth):
                    pipe_ind = 1
        else:
            break

        draw_window(win, birds, pipes, generation,ge)

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].get_first_pipe()),
                                       abs(bird.y - pipes[pipe_ind].get_second_pipe())))

            if (output[0] > 0.5):
                bird.jump()

        for bird in birds:
            if (colide(pipes[pipe_ind], bird)):
                ge[birds.index(bird)].fitness -= 1
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        for x, pipe in enumerate(pipes):
            if (pipe.passed(bird) == True):
                for genome in ge:
                    genome.fitness += 5
                pipe = Pipe()
                pipes.pop(x)
                pipes.append(pipe)


def run(config_file):
    
    """
        runs the NEAT algorithm to train a neural network to play flappy bird.
        :param config_file: location of config file
        :return: None
        """
   
   # config = neat.config.Config(genome_type, reproduction_type, species_set_type, stagnation_type, filename)(neat.DefaultGenome, neat.DefaultReproduction,
   #                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
   #                             config_file)
   
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(main, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


def configurate_file():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'nn.txt')
    run(config_path)





configurate_file()
