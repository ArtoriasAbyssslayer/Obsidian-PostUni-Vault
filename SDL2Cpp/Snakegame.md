```c
#include <SDL.h>
#include <vector>
#include <algorithm>
#include <deque>
#include <iostream>
int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_EVERYTHING);
    auto window = SDL_CreateWindow("SnakeGame", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1000, 1000, 0);
    auto renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Event e;
    bool gameRunning = true;

    enum SnakeDirection {
        DOWN,
        LEFT,
        RIGHT,
        UP
    };


    SDL_Rect snakeHead = { 500, 500, 10, 10 };
    int dir = 0;
    std::deque<SDL_Rect> snakeBodyRectQueue;
    int size = 1;

    std::vector<SDL_Rect> applesVectPos;
    applesVectPos.push_back({ rand() % 100 * 10, rand() % 100 * 10, 10, 10 });
    while (gameRunning) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                gameRunning = false;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_DOWN) {
                    dir = DOWN;
                }
                if (e.key.keysym.sym == SDLK_UP) {
                    dir = UP;
                }
                if (e.key.keysym.sym == SDLK_RIGHT) {
                    dir = RIGHT;
                }
                if (e.key.keysym.sym == SDLK_LEFT) {
                    dir = LEFT;
                }
            }
        }

        switch (dir) {
        case DOWN:
            snakeHead.y += 10;
            if (snakeHead.y >= 1000) {
                snakeHead.y = 0;
            }
            break;
        case UP:
            snakeHead.y -= 10;
            if (snakeHead.y < 0) {
                snakeHead.y = 1000; // Adjust based on the size of the snake
            }
            break;
        case RIGHT:
            snakeHead.x += 10;
            if (snakeHead.x >= 1000) {
                snakeHead.x = 0;
            }
            break;
        case LEFT:
            snakeHead.x -= 10;
            if (snakeHead.x <= 0) {
                snakeHead.x = 1000;
            }
            break;
        }

        std::for_each(applesVectPos.begin(), applesVectPos.end(), [&](auto& apple) {
            if (snakeHead.x == apple.x && snakeHead.y == apple.y) {
                size += 5;
                apple.x = -10;
                apple.y = -10;
                applesVectPos.push_back({ rand() % 100 * 10, rand() % 100 * 10, 10, 10 });
            }
        });

        std::for_each(snakeBodyRectQueue.begin(), snakeBodyRectQueue.end(), [&](auto& bodySegment) {
            if (snakeHead.x == bodySegment.x && snakeHead.y == bodySegment.y) {
                size = 1;
    
                //gameRunning = false;
            }
        });

        snakeBodyRectQueue.push_front(snakeHead);
        while (size <= snakeBodyRectQueue.size()) {
            snakeBodyRectQueue.pop_back();
        }
        //Clear the Window
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        std::for_each(snakeBodyRectQueue.begin(), snakeBodyRectQueue.end(), [&](auto& snake_segment) {
            SDL_RenderFillRect(renderer, &snake_segment);
        });
        SDL_RenderFillRect(renderer, &snakeHead);
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        std::for_each(applesVectPos.begin(), applesVectPos.end(), [&](auto& apple) {
			SDL_RenderFillRect(renderer, &apple);
		});
        //SDL_RenderFillRect(renderer, &applesVectPos[++k]);               
        SDL_RenderPresent(renderer);
        int k = 50;
        SDL_Delay(k-=5);
    }

    return 0;
}

```

