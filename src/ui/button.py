import pygame


class Button:
    def __init__(self, text, dims, rel_pos, on_click, offset=(0, 0)):
        self.bg_color = (150, 150, 150)
        self.text = text
        self.dims = dims
        self.offset = offset
        self.on_click = on_click
        self.rel_pos = rel_pos
        pygame.font.init()
        font = pygame.font.Font(None, size=20)
        self.rendered_text = font.render(self.text, True, (0, 0, 0))
        pygame.font.quit()

    def get_abs_pos(self):
        rel_x, rel_y = self.rel_pos
        offset_x, offset_y = self.offset
        abs_x = rel_x + offset_x
        abs_y = rel_y + offset_y
        result = (abs_x, abs_y)
        return result

    def mouse_is_above(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_x, mouse_y = mouse_pos
        button_x, button_y = self.get_abs_pos()
        width, height = self.dims
        in_domain = 0 <= mouse_x - button_x <= width
        in_range = 0 <= mouse_y - button_y <= height
        return in_range and in_domain

    def listen(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.mouse_is_above():
                    self.on_click()
                    return True
        return False

    def draw(self, surface):
        button_surface = pygame.Surface(self.dims)
        button_surface.fill(self.bg_color)
        button_surface.blit(self.rendered_text, (0, 0))
        surface.blit(button_surface, self.rel_pos)
