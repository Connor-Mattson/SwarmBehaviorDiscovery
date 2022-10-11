import pygame
import numpy as np
import cv2

class SimilarityGUI:
    def __init__(self, anchor, subjects):
        self.subject_s = None
        self.anchor_s = None

        self.WIDTH = 1600
        self.HEIGHT = 1000
        self.IMG_SIZE = (250, 250)
        self.CONTAINER_SIZE = (300, 300)

        self.anchor = anchor
        self.subjects = subjects
        self.makeSurfaces()
        self.selected = 1
        self.assignment = [None for _ in subjects]

        pygame.init()
        pygame.display.set_caption("Swarm Behavior Classification")
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def resize(self, img, size=(200, 200)):
        return cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

    def makeSurfaces(self):
        self.anchor_s = pygame.surfarray.make_surface(self.resize(self.anchor, size=self.IMG_SIZE))
        self.subject_s = [pygame.surfarray.make_surface(self.resize(i, size=self.IMG_SIZE)) for i in self.subjects]

    def getIMGLocation(self, i):
        if i == 0:
            return (self.WIDTH / 2) - (self.CONTAINER_SIZE[0] / 2), 10
        else:
            return (self.CONTAINER_SIZE[0] * ((i - 1) % 5)), (((i - 1) // 5) + 1) * self.CONTAINER_SIZE[1]

    def getOppositeCorner(self, point):
        return point[0] + self.CONTAINER_SIZE[0], point[1] + self.CONTAINER_SIZE[1]

    def showImg(self, img, i):
        pos = self.getIMGLocation(i)
        if self.selected == i:
            pygame.draw.rect(self.screen, (255, 255, 255), (pos, self.CONTAINER_SIZE))

        self.screen.blit(img, (pos[0] + 25, pos[1] + 25))

        if self.assignment[i - 1] is not None or i == 0:
            if i == 0:
                text = self.font.render("Anchor", True, (0, 0, 255), (255, 255, 255))
            else:
                caption = "Same" if self.assignment[i - 1] == 0 else "Different"
                text = self.font.render(caption, True, (255, 0, 0), (255, 255, 255))
            textRect = text.get_rect()
            textRect.x = pos[0]
            textRect.y = pos[1]
            self.screen.blit(text, textRect)

    def clickInGUI(self, click):
        click_point = np.array(click)
        pass

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    self.clickInGUI(pos)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.selected = min(self.selected + 1, len(self.subjects))
                    if event.key == pygame.K_LEFT:
                        self.selected = max(self.selected - 1, 1)
                    if event.key == pygame.K_KP1:
                        self.assignment[self.selected - 1] = 1
                    if event.key == pygame.K_KP0:
                        self.assignment[self.selected - 1] = 0
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        running = False

            self.screen.fill((0, 0, 0))
            self.showImg(self.anchor_s, 0)
            for i, img in enumerate(self.subject_s):
                self.showImg(img, i + 1)

            pygame.display.update()

        pygame.quit()