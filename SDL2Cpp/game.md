```cpp
#include <Game.h>
#include <cstdlib>
#include <cmath>
#include <Player.h>

SDL_Texture* enemySprite;
Game::Game() : s_Window(nullptr), s_Renderer(nullptr), game_over(false) {
	// Initialize SDL and other resources here
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		// Handle initialization error
		// You might want to throw an exception or set game_over to true
		game_over = true;
	}

	// Initialize the window and renderer
	s_Window = SDL_CreateWindow("Interconnected Souls", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
	if (!s_Window) {
		// Handle window creation error
		game_over = true;
	}

	s_Renderer = SDL_CreateRenderer(s_Window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (!s_Renderer) {
		// Handle renderer creation error
		game_over = true;

	}
    Player(s_Renderer,"assets/SwordmanSprite.png", )

}

Game::~Game() {
	// Clean up SDL and other resources here
	if (s_Renderer) {
		SDL_DestroyRenderer(s_Renderer);
	}

	if (s_Window) {
		SDL_DestroyWindow(s_Window);
	}

	SDL_Quit();
}

void Game::handle_events()
{
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0) {

        Player.handle_input();
        Enemy.AI();
        /*Events for game Close*/
        if (e.type == SDL_QUIT) {
            printf("Quit event detected\n");
            game_over = true;
        } else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_CLOSE) {
            printf("Window close event detected\n");
            game_over = true;
        }
    }
}
void Game::init(const char* game_title, int x_pos, int y_pos, int canvas_width, int canvas_height, int fullscreen_flag)
{
	//attempt to initialilze sdl
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0)
	{
		std::cout << "SDL_initialized" << std::endl;

		// init window
		s_Window = SDL_CreateWindow(game_title, x_pos, y_pos, canvas_width, canvas_height, fullscreen_flag);

		if (s_Window) // window entity created
		{
			std::cout << "Game Window Created\n";
			s_Renderer = SDL_CreateRenderer(s_Window, -1, 0);

			if(!s_Renderer)
			{
#				//TODO: add error handling
				std::cout << "Renderer creation failed" << std::endl;
			}
			else
			{
				// Draw a white background
				std::cout << "Renderer created" << std::endl;
				SDL_SetRenderDrawColor(s_Renderer, 0x19, 0x19, 0x19, 0x04);
			}
		}
	}

}



void Game::update() {
	counter++;
	//std::cout << counter << "\n";
}

void Game::render() {
    // Implement rendering logic
    SDL_RenderClear(s_Renderer);
    // Add rendering code here
    //Render Enemy
    SDL_RenderCopy(s_Renderer, enemySprite, NULL, NULL);
    // RenderPlayer
    Player.render(s_Renderer);
    SDL_RenderPresent(s_Renderer);
}


void Game::clean()
{

    SDL_DestroyTexture(enemySprite);
    SDL_DestroyRenderer(s_Renderer);
    SDL_DestroyWindow(s_Window);
    IMG_Quit();
    SDL_Quit();
}

```

## Main 
```cpp
#include <Game.h>

Game *game = nullptr;

int WinMain(int argc, char* argv[])
{
	const int FPS = 60;
	const int frameDelay = 1000 / FPS;

	Uint32 frameStart;
	int frameTime;

	game = new Game();
	game->init("Interconnected Souls", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
  std::cout<<"Game Running: "<<std::boolalpha<<game->running()<<std::endl;
  while (game->running())
	{
        frameStart = SDL_GetTicks();
		game->handle_events();
		game->update();
		game->render();
		frameTime = SDL_GetTicks() - frameStart;
		if (frameDelay > frameTime)
		{
			SDL_Delay(frameDelay - frameTime);

		}
	}
  delete game;
  return 0;
}

```
