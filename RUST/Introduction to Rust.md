

* Rust is a multiparadigm, compiled programming language that developers can view as a modern version of C and C++. 
* It is a *statically* and **strongly-typed** *functional language*
* Rust uses syntax similar to C++ AND PROVIDES safety-first principles to ensure programmers write stable and extendable, asynchronous code 




# Hello World

### Creating a project
```sh
$ mkdir -p ~/projects/hello_world && cd ~/projects/hello_world 
```
### Main.rs
```rust
fn main(){
	println!("Hello, world!");
}
```
### Compile and run 
```sh
$ rustc main.rs
$ ./main 
Hello, world!
```
