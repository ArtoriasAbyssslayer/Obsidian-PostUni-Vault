
### DAY 01.html
```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <title>Day 01</title>
</head>

<body>
    <h1>Setting Up PHP</h1>

    <h2>Summary</h2>

    <blockquote>
        <ul>
            <li>PHP is a server-side scripting language used for web development.</li>
            <li>Setting up PHP involves installing it on your operating system and configuring it with a web server.
            </li>
            <li>This guide covers setup instructions for Windows, macOS, Ubuntu, Fedora, and other Linux distributions.
            </li>
        </ul>
    </blockquote>

    <h2>Instructions by Operating System</h2>

    <h3>Windows</h3>
    <blockquote>
        <h2>Prerequisites</h2>

        <ul>
            <li>A Windows computer</li>
            <li>An internet connection</li>
            <li>Administrator privileges on your computer</li>
        </ul>

        <h2>Steps</h2>

        <h3>1. Download PHP</h3>

        <ul>
            <li>Visit the official PHP website: <a
                    href="https://www.php.net/downloads.php">https://www.php.net/downloads.php</a></li>
            <li>Download the latest stable Windows version that's compatible with your system (x86 or x64).</li>
            <li>Choose the "Thread Safe".</li>
            <li>Save the downloaded file (e.g., "php-8.2.13-Win32-vs16-x64.zip").</li>
        </ul>

        <h3>2. Extract PHP</h3>

        <ul>
            <li>Extract the downloaded ZIP file to a suitable location, such as "C:\php".</li>
        </ul>

        <h3>3. Add PHP to PATH</h3>

        <ul>
            <li>Right-click "This PC" or "My Computer" and select "Properties."</li>
            <li>Click "Advanced system settings."</li>
            <li>Click "Environment Variables."</li>
            <li>Under "System variables," find "Path" and click "Edit."</li>
            <li>Click "New" and add the path to your PHP directory (e.g., "C:\php").</li>
            <li>Click "OK" on all open dialog boxes to save changes.</li>
        </ul>

        <h3>4. Test PHP Installation</h3>

        <ul>
            <li>Open a command prompt (search for "cmd" in the Start menu).</li>
            <li>Type "php -v" and press Enter.</li>
            <li>If PHP is installed correctly, you should see the PHP version information.</li>
        </ul>
    </blockquote>

    <h3>macOS</h3>
    <blockquote>
        <h2>Prerequisites</h2>

        <ul>
            <li>A macOS computer</li>
            <li>An internet connection</li>
            <li>Administrator privileges on your computer</li>
        </ul>

        <h2>Steps</h2>

        <h3>1. Install Homebrew (if not already installed)</h3>

        <ul>
            <li>Open Terminal (located in Applications > Utilities).</li>
            <li>Paste the following command and press Enter:
                <pre><code>/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"</code></pre>
            </li>
            <li>Follow the on-screen instructions to complete the Homebrew installation.</li>
        </ul>

        <h3>2. Install PHP using Homebrew</h3>

        <ul>
            <li>In Terminal, type the following command and press Enter:
                <pre><code>brew install php</code></pre>
            </li>
        </ul>

        <h3>3. Test PHP Installation</h3>

        <ul>
            <li>In Terminal, type <code>php -v</code> and press Enter.</li>
            <li>If PHP is installed correctly, you should see the PHP version information.</li>
        </ul>
    </blockquote>


    <h3>Ubuntu</h3>
    <blockquote>
        <h2>Prerequisites</h2>

        <ul>
            <li>An Ubuntu computer</li>
            <li>An internet connection</li>
            <li>Administrator privileges (use the "sudo" command)</li>
        </ul>

        <h2>Steps</h2>

        <h3>1. Update Package Lists</h3>

        <p>To ensure you have the latest package information, open a terminal window (Ctrl+Alt+T) and run:</p>
        <pre><code>sudo apt update</code></pre>

        <h3>2. Install PHP</h3>

        <p>To install PHP and its common extensions, run:</p>
        <pre><code>sudo apt install php libapache2-mod-php php-mysql php-curl php-gd php-mbstring php-xml php-xmlrpc php-soap php-intl php-zip</code></pre>

        <h3>3. Verify Installation</h3>

        <p>To check if PHP is installed correctly, run:</p>
        <pre><code>php -v</code></pre>
        <p>This should display the PHP version information.</p>

        <h3>4. Restart Apache (if applicable)</h3>

        <p>If you're using Apache, restart it for the changes to take effect:</p>
        <pre><code>sudo systemctl restart apache2</code></pre>

        <h3>5. Test PHP Functionality</h3>

        <p>Create a simple PHP file named "info.php" within your web root directory (usually "/var/www/html") with the
            following content:</p>
        <pre><code>&lt;?php
phpinfo();
?&gt;</code></pre>
        <p>Access this file in your web browser (e.g., http://localhost/info.php). It should display detailed
            information about your PHP configuration.</p>

    </blockquote>

    <h3>Other Linux Distributions</h3>
    <blockquote>
        <ul>
            <li><b>Fedora:<a href="https://m.youtube.com/watch?v=b4c1DjJOT2M">Install PHP on Fedora Linux 37 (Easy
                        Guide)</a> by microcodes</li>
            <li><b>Arch Linux:<a href="https://m.youtube.com/watch?v=nrCuTiIQuOk">Arch Linux - Installing PHP 8.1 and
                        Apache 2.4 for Web Development</a> by Quick Notepad Tutorial
                    </li>
            <li><b>Mint:<a href="https://www.youtube.com/watch?v=fXhlSvWSz5U">How to Install PHP on Linux Mint 21 |
                        XAMPP vs LAMP | Server-Side Scripting</a> by EaseCoding</li>
            <li><b>Other Distros:<a href="https://m.youtube.com/watch?v=6kAr7VRuh9w">How to Install PHP on Any Linux
                        Distro 2023 | LAMP Stack Tutorial</a> by ATOM</li>
        </ul>
    </blockquote>
    <div class="footer">
        <a href="/">Home</a> | 
        <a href="/Day/Day 02.php"> Day 02</a>
    </div>
</body>

</html>
```
#### Day 02.php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <title>Day 02</title>
</head>

<body>
    <h1>Day 02</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li><code>#this is a Comment</code> and <code>//this is a Comment</code> are used to create single-line
                comments.</li>
            <li><code>/* this is a multi-line Comment */</code> is used for multi-line comments.</li>
            <li><code>echo</code> allows multiple parameters to be displayed and has no return value (e.g.,
                <code>echo "this", "is", "an output";</code>).
            </li>
            <li><code>print</code> allows only one parameter and returns "1" (e.g., <code>print "HI!";</code>,
                <code>print "Hi";</code>).
            </li>
            <li>PHP is a dynamic language, so there's no need to declare variable types.</li>
            <li>Variables in PHP start with a <code>$</code> sign.</li>
            <li><a href="https://www.w3schools.com/php/php_variables.asp">PHP Variables Rules</a></li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
    # This is a Comment
    // This is Also A Comment

    /*
    This is a multiline
    comment
    */

    echo "this ", "is ", "an output";

    print "HI!";
    // print "Hi", "This is not gonna work";

    $_txt = "PHP";
    echo "I love $_txt!";
    </code></pre>
    </blockquote>
    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            # This is a Comment
            // This is Also A Comment
            
            /*
            This is a multiline
            comment
            */

            echo "this ", "is ", "an output", "<br>";

            print "HI!";
            // print "Hi", "This is not gonna work";
            
            $_txt = "PHP";
            echo "I love $_txt!";
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 01.html"> Day 01</a>
         | <a href="/">Home</a> | 
        <a href="/Day/Day 03.php"> Day 03</a>
    </div>
</body>

</html>
```
#Day03.php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <title>Day 03</title>
</head>

<body>
    <h1>Day 03</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>Arithmetic Operators [*,-,*,/,%,**]</li>
            <li>Assignment Operators [=, +=, -=, *=, /=, %=]</li>
            <li>Constant</li>
            <li>The ability to define case-insensitive constants in PHP was deprecated in version 7.3 and completely
                removed in version 8.0. </li>
            <li>Array</li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
    // Arithmetic Operators [*,-,*,/,%,**] (Removed Text Formating)
    echo "This is Addition of 22 & 23:- ",22+23;
    echo "This is Subtraction of 22 & 23:- ",22-23;
    echo "This is Multiplication of 22 & 23:- ",22*23;
    echo "This is Division of 22 & 23:- ",22/23;
    echo "This is Modulus of 22 & 23:- ",22%23;
    echo "This is Exponentiation of 22 & 23:- ",22**23;
    echo "This is Integer Division of 22 & 23", intdiv(22,26);

    // Assignment Operators [=, +=, -=, *=, /=, %=] (Removed Text Formating)
    $x = 10; echo "x = ", $x;
    $x += 5; echo "x+=5 = ", $x;
    $x -= 5; echo "x-=5 = ", $x;
    $x *= 5; echo "x*=5 = ", $x;
    $x /= 5; echo "x/=5 = ", $x;
    $x %= 5; echo "x%=5 = ", $x;
    $x **= 5; echo "x**=5 = ", $x;

    //Constant (Case Sensitive)
    define("CC","HI! I'm Constent. I Can Be A Integer or String or Anything");

    //Array
    define("ImArray", ["Hi!, ","I am ", "a Array."]);
    echo ImArray[0], ImArray[1], ImArray[2];
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            // Arithmetic Operators [*,-,*,/,%,**]
            echo "This is Addition of 22 & 23:- <b>", 22 + 23, "</b><br />";
            echo "This is Subtraction of 22 & 23:- <b>", 22 - 23, "</b><br />";
            echo "This is Multiplication of 22 & 23:- <b>", 22 * 23, "</b><br />";
            echo "This is Division of 22 & 23:- <b>", 22 / 23, "</b><br />";
            echo "This is Modulus of 22 & 23:- <b>", 22 % 23, "</b><br />";
            echo "This is Exponentiation of 22 & 23:- <b>", 22 ** 23, "</b><br />";
            echo "This is Integer Division of 22 & 23:- <b>", intdiv(22, 23), "</b><br />";
            echo "<br />";

            // Assignment Operators [=, +=, -=, *=, /=, %=]
            $x = 10;
            echo "x = <b>", $x, "</b><br />";
            $x += 5;
            echo "x+=5 = <b>", $x, "</b><br />";
            $x -= 5;
            echo "x-=5 = <b>", $x, "</b><br />";
            $x *= 5;
            echo "x*=5 = <b>", $x, "</b><br />";
            $x /= 5;
            echo "x/=5 = <b>", $x, "</b><br />";
            $x %= 5;
            echo "x%=5 = <b>", $x, "</b><br />";
            $x **= 5;
            echo "x**=5 = <b>", $x, "</b><br />";
            echo "<br />";

            //Constant (Case Sensitive)
            define("CC", "HI! I'm Constent. I Can Be A Integer or String or Anything");
            echo CC, "<br />";
            echo "<br />";

            //Array
            define("ImArray", ["Hi!, ", "I am ", "a Array."]);
            echo ImArray[0], ImArray[1], ImArray[2];
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 02.php"> Day 02</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 04.php"> Day 04</a>
    </div>
</body>

</html>
```
#day04.php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <title>Day 04</title>
</head>

<body>
    <h1>Day 03</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>Data Types</li>
            <li>var_dump function used to display data type and the data of a variable</li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
        /* String (Removed Text Formmating)
        A combinations of Unicode
        Use Single or Double Quotes*/
        $String = "Hi, I am String; I can be Anything.
        බලන්න, මට ඕනෑම දෙයක් විය හැකිය!
        ほら、私は何にでもなれるよ！
        Δείτε, μπορώ να είμαι οτιδήποτε!
        பார், நான் எதுவும் ஆக முடியும்!";
        var_dump($String);


        /*Integer (Removed Text Formmating)
        A combinations of Whole Numbers (0-9)*/
        $Integer = -4540356805;
        var_dump($Integer);


        /*Float (Removed Text Formmating)
        A number with a decimal point or a number in exponential form*/
        $Float = -45403.56805;
        var_dump($Float);

        /*Boolean (Removed Text Formmating)
        True or False*/
        $Boolean = true;
        var_dump($Boolean);
        $Boolean = false;
        var_dump($Boolean);


        /*Arrays
        Combinations of Anything*/
        $Array = [3,3.1415,"π",true,["The number π is a mathematical constant that ... approximately equal to 3.14159.", "https://en.wikipedia.org/wiki/Pi", [3.141507816406286, "http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html"]]];
        var_dump($Array);  
        echo $Array[4][2][0];
        $Array[4][2][0] = 3.1415;
        echo $Array[4][2][0];
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            /*
            String
            A combinations of Unicode
            Use Single or Double Quotes
            */
            $String = "Hi, I am String; I can be Anything.<br />බලන්න, මට ඕනෑම දෙයක් විය හැකිය! <br /> ほら、私は何にでもなれるよ！ <br /> Δείτε, μπορώ να είμαι οτιδήποτε! <br /> பார், நான் எதுவும் ஆக முடியும்!";
            var_dump($String);
            echo "<br /> <br />";


            /*
            Integer
            A combinations of Whole Numbers (0-9)
            */
            $Integer = -4540356805;
            var_dump($Integer);
            echo "<br /> <br />";


            /*
            Float
            A number with a decimal point or a number in exponential form
            */
            $Float = -45403.56805;
            var_dump($Float);
            echo "<br /> <br />";


            /*
            Boolean
            True or False
            */
            $Boolean = true;
            var_dump($Boolean);
            print "<br />";
            $Boolean = false;
            var_dump($Boolean);       
            echo "<br /> <br />";


            /*
            Arrays
            Combinations of Anything
            */
            $Array = [3,3.1415,"π",true,["The number π is a mathematical constant that ... approximately equal to 3.14159.", "https://en.wikipedia.org/wiki/Pi", [3.141507816406286, "http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html"]]];
            var_dump($Array);  
            print("<br />");
            echo $Array[4][2][0], "<br />";
            $Array[4][2][0] = 3.1415;
            echo $Array[4][2][0];
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 03.php"> Day 03</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 05.php"> Day 05</a>
    </div>
</body>

</html>
```
#Day05.php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <title>Day 05</title>
</head>

<body>
    <h1>Day 05</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>Conditonal Statements
                <ul>
                    <li>if</li>
                    <li>if else</li>
                    <li>if elseif else</li>
                    <li>switch</li>
                </ul>
            </li>
            <li>PHP Comparison Operators [==, ===, !=, <>, !==, >, <,>=, <=]</li>
            <li>Spaceship; A New PHP Comparison Operators [<=>]</li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
        // If Statement
        $t = 5;
        if ($t == 5) {
            echo "Hi! I am if.";
        }
        
        // If elseif else statement
        $y = 6;
        if ($y == 5) {
            echo "Hi! I am if.";
        } elseif ($y == 6) {
            echo "Hi! elseif.";
        } else {
            echo "Hi! I am else";
        }


        //switch case statement
        //Copied from: https://www.w3schools.com/php/phptryit.asp?filename=tryphp_switch
        $favcolor = "red";
        switch ($favcolor) {
            case "red":
                echo "Your favorite color is red!";
                break;
            case "blue":
                echo "Your favorite color is blue!";
                break;
            case "green":
                echo "Your favorite color is green!";
                break;
            default:
                echo "Your favorite color is neither red, blue, nor green!";
        }

        // PHP Comparison Operators
        $x = 9;
        $y = "9";
        echo "Equal; ", var_dump($x == $y);
        echo "Identical; ", var_dump($x === $y)";
        echo "Not equal(!=); ", var_dump($x != $y);
        echo "Not equal(<>); ", var_dump($x <> $y);
        echo "Not Identical; ", var_dump($x !== $y);
        echo "Greater than or equal to; ", var_dump($x >= $y);

        // Spaceship; A New PHP Comparison Operators
        $x = 51;
        $y = 101;
        echo ($x <=> $y);

        $x = 101;
        $y = 101;
        echo ($x <=> $y);

        $x = 151;
        $y = 101;
        echo ($x <=> $y);
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            // If Statement
            $t = 5;
            if ($t == 5) {
                echo "Hi! I am if.";
            }
            echo "<br />";

            // If elseif else statement
            $y = 6;
            if ($y == 5) {
                echo "Hi! I am if.";
            } elseif ($y == 6) {
                echo "Hi! elseif.";
            } else {
                echo "Hi! I am else";
            }
            echo "<br />";
            //switch case statement
            //Copied from: https://www.w3schools.com/php/phptryit.asp?filename=tryphp_switch
            $favcolor = "red";

            switch ($favcolor) {
                case "red":
                    echo "Your favorite color is red!";
                    break;
                case "blue":
                    echo "Your favorite color is blue!";
                    break;
                case "green":
                    echo "Your favorite color is green!";
                    break;
                default:
                    echo "Your favorite color is neither red, blue, nor green!";
            }

            echo "<br />", "<br />";

            // PHP Comparison Operators
            $x = 9;
            $y = "9";
            echo "Equal; ", var_dump($x == $y), "<br />";
            echo "Identical; ", var_dump($x === $y), "<br />";
            echo "Not equal(!=); ", var_dump($x != $y), "<br />";
            echo "Not equal(<>); ", var_dump($x <> $y), "<br />";
            echo "Not Identical; ", var_dump($x !== $y), "<br />";
            echo "Greater than or equal to; ", var_dump($x >= $y), "<br />";
            echo "<br />", "<br />";

            // Spaceship; A New PHP Comparison Operators
            $x = 51;
            $y = 101;
            echo ($x <=> $y);
            echo "<br>";
            $x = 101;
            $y = 101;
            echo ($x <=> $y);
            echo "<br>";
            $x = 151;
            $y = 101;
            echo ($x <=> $y);
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 04.php"> Day 04</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 06.php"> Day 06</a>
    </div>
</body>

</html>
```
#Day06 .php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/main.css">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <title>Day 06</title>
    <style>
        .spoiler:hover {
            color: black;
        }

        .spoiler {
            color: #cecece00;
        }
    </style>
</head>

<body>
    <h1>Day 06</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>Repetition Control Structures
                <ul>
                    <li>while</li>
                    <li>do..while</li>
                    <li>for</li>
                    <li>foreach</li>
                </ul>
            </li>
            <li>Break & Continue</li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
    //While Loop
    $n = 1;
    while ($n < 6) {
        echo "Hi! I am ", $n;
        $n += 1;
    }


    // Do While Loop (Copied From: https://www.w3schools.com/php/php_looping_do_while.asp)
    $i = 1;
    do {
        echo $i;
        $i++;
    } while ($i < 6);


    //for loop (Copied From: https://www.w3schools.com/php/php_looping_for.asp)
    for ($x = 0; $x <= 10; $x++) {
        echo "The number is: $x";
    }


    //for each loop (Copied From: https://www.w3schools.com/php/php_looping_foreach.asp)
    $colors = array("red", "green", "blue", "yellow");
    foreach ($colors as $x) {
        echo "$x";
    }

    //Break
    for ($p = 0; $p < 10; $p++) {
        if ($p == 5) {
            echo "Okay, I will Break Here; ", $p;
            break;
        }
        echo "HI, I am $p";
    }
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            //While Loop
            $n = 1;
            while ($n < 6) {
                echo "Hi! I am ", $n, "<br />";
                $n += 1;
            }
            echo "<br />", "<br />";

            // Do While Loop (Copied From: https://www.w3schools.com/php/php_looping_do_while.asp)
            $i = 1;

            do {
                echo $i;
                $i++;
            } while ($i < 6);
            echo "<br />", "<br />";

            //for loop (Copied From: https://www.w3schools.com/php/php_looping_for.asp)
            for ($x = 0; $x <= 10; $x++) {
                echo "The number is: $x", "<br>";
            }
            echo "<br />", "<br />";

            //for each loop (Copied From: https://www.w3schools.com/php/php_looping_foreach.asp)
            $colors = array("red", "green", "blue", "yellow");

            foreach ($colors as $x) {
                echo "$x", " <br>";
            }
            echo "<br />", "<br />";

            //Break
            for ($p = 0; $p < 10; $p++) {
                if ($p == 5) {
                    echo "<b>", "Okay, I will Break Here; ", $p, "</b>";
                    break;
                }
                echo "HI, I am $p <br />";
            }
            echo "<br />", "<br />";

            //Continue
            for ($p = 0; $p < 10; $p++) {
                if ($p == 5 or $p == 6) {
                    echo "<span class='spoiler'>", "Okay, I will Be Not Display ", $p, "</span><br />";
                    continue;
                }
                echo "HI, I am $p <br />";
            }
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 05.php"> Day 05</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 07.php"> Day 07</a>
    </div>
</body>

</html>
```

#Day07 .php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <link rel="stylesheet" href="/main.css">
    <title>Day 07</title>
</head>

<body>
    <h1>Day 07</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>String Functions
                <ul>
                    <li>strlen</li>
                    <li>str_word_count</li>
                    <li>strrev</li>
                    <li>strpos</li>
                    <li>str_replace</li>
                </ul>
            </li>
            <li>User Defined Functions
                <ul>
                    <li>No Parameters, No Return</li>
                    <li>With Parameters, No Return</li>
                    <li>No Parameters, With Return</li>
                    <li>With Parameters, With Return</li>
                </ul>
            </li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
    <h3>String Functions</h3>

    //strlen(String)
    echo "Count Me: ", strlen("Count Me");

    //str_word_count()
    echo "Count The Words, Actualy Word with spaces ",
    str_word_count("Count The Words, Actualy Word with spaces : ");

    //strrev
    echo "I am in reverse | ", strrev("I am in reverse")";

    //strpos
    echo "Lorem, ipsum dolor sit amet consectetur adipisicing elit.
    Consequuntur, voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis
    assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor.String position
    of 'voluptas cupiditate' is ", strripos("Lorem, ipsum dolor sit amet consectetur 
    adipisicing elit. Consequuntur, voluptas cupiditate neque hic quis eligendi fuga 
    quam nam blanditiis assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet
    dolor.", "voluptas cupiditate");

    //str_replace()
    echo "Lorem, ipsum dolor sit amet consectetur adipisicing elit. Consequuntur, 
    voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis assumenda ipsa
    nemo magnam voluptatem sunt sequi enim, at amet dolor. | "
    , str_replace("ipsum dolor sit amet consectetur adipisicing elit.
    Consequuntur"
    , "I'm replaced", "Lorem, ipsum dolor sit amet consectetur adipisicing elit. 
    Consequuntur,voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis 
    assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor.");

    <h3>User Defined Functions</h3>

    // No Parameters, No Return
    function hi()
    {
        $text = hello();
        frame($text);
        echo "$text";
        frame($text);
    }
    ;
    hi();

    // With Parameters, No Return
    function frame($text)
    {
        for ($i = 0; $i < strlen($text); $i++) {
            print "#";
        }
        ;
    }

    //No Parameters, With Return
    function hello()
    {
        return "Hi, How Are You?";
    }
    ;

    // With Parameters, With Return (Copied:https://www.w3schools.com/php/php_functions.asp)
    function sum($x, $y)
    {
        $z = $x + $y;
        return $z;
    }
    echo "5 + 10 = " . sum(5, 10);
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            echo "<h3>String Functions</h3>";

            //strlen(String)
            echo "Count Me: ", strlen("Count Me"), "<br /> <br />";

            //str_word_count()
            echo "Count The Words, Actualy Word with spaces ", str_word_count("Count The Words, Actualy Word with spaces : "), "<br /> <br />";

            //strrev
            echo "I am in reverse | ", strrev("I am in reverse"), "<br /> <br />";

            //strpos
            echo "Lorem, ipsum dolor sit amet consectetur adipisicing elit. Consequuntur, voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor. <br> <b>String position of 'voluptas cupiditate' is </b>", strripos("Lorem, ipsum dolor sit amet consectetur adipisicing elit. Consequuntur, voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor.", "voluptas cupiditate"), "<br /><br />";

            //str_replace()
            echo "Lorem, ipsum dolor sit amet consectetur adipisicing elit. Consequuntur, voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor. | <b>", str_replace("ipsum dolor sit amet consectetur adipisicing elit. Consequuntur", "I'm replaced", "Lorem, ipsum dolor sit amet consectetur adipisicing elit. Consequuntur, voluptas cupiditate neque hic quis eligendi fuga quam nam blanditiis assumenda ipsa nemo magnam voluptatem sunt sequi enim, at amet dolor."), "</b><br /><br />";

            echo "<br>", "<h3>User Defined Functions</h3>";
            // No Parameters, No Return
            function hi()
            {
                $text = hello();
                frame($text);
                echo "<br> $text <br>";
                frame($text);
            }
            ;
            hi();
            // With Parameters, No Return
            function frame($text)
            {
                for ($i = 0; $i < strlen($text); $i++) {
                    print "#";
                }
                ;
            }

            //No Parameters, With Return
            function hello()
            {
                return "Hi, How Are You?";
            }
            ;

            // With Parameters, With Return (Copied:https://www.w3schools.com/php/php_functions.asp)
            function sum($x, $y)
            {
                $z = $x + $y;
                return $z;
            }

            echo "<br><br> 5 + 10 = " . sum(5, 10) . "<br>";
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 06.php"> Day 06</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 08.php"> Day 08</a>
    </div>
</body>

</html>
```

#Day08 .php
```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <link rel="stylesheet" href="/main.css">
    <title>Day 08</title>
</head>

<body>
    <h1>Day 08</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
            <li>Variable Scope
                <ul>
                    <li>Global Scope</li>
                    <li>Local Scope</li>
                    <li>Static Scope</li>
                </ul>
            </li>
            <li>Global Keywords</li>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>

    //Global Scope
    $var = "Variable Of Global Scope.";
    function testFun()
    {
        // Local Scope
        $var = "Variable Of Local Scope.";
        echo "I Am A " . $var;
        //Global Variable Keyword
        global $var;
        echo "But I can Access Global Varaiables With The 'Global Keyword'. Look I'm $var";
        echo "Or I can Get Global Var With $ GLOBALS[index]; Look, I am still $GLOBALS[var];
    }
    testFun();
    echo "Look even though Varaiable changed in testFun, I never changed, cause I'm $var";
    //The Static Keyword (Copied)
    function add1()
    {
        static $number = 0; // Try Removing 'static' keyword
        $number++;
        return $number;
    }
    echo add1();
    echo add1();
    echo add1();
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php

            //Global Scope
            $var = "Variable Of Global Scope.";

            function testFun()
            {
                // Local Scope
                $var = "Variable Of Local Scope.";
                echo "I Am A " . $var . "<br >";

                //Global Variable Keyword
                global $var;
                echo "But I can Access Global Varaiables With The 'Global Keyword'. Look I'm $var <br><br>";
                echo "Or I can Get Global Var With $ GLOBALS[index]; Look, I am still $GLOBALS[var] <br>";
            }
            testFun();
            echo "<br>Look even though Varaiable changed in testFun, I never changed, cause I'm $var <br><br>";

            //The Static Keyword (Copied)
            function add1()
            {
                static $number = 0; // Try Removing 'static' keyword
                $number++;
                return $number;
            }

            echo add1();
            echo "<br>";
            echo add1();
            echo "<br>";
            echo add1();
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 07.php"> Day 07</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 09.php"> Day 09</a>
    </div>
</body>

</html>
```



# day_template.php

```php
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
    <link rel="stylesheet" href="/main.css">
    <title>Day 03</title>
</head>

<body>
    <h1>Day 03</h1>
    <h4>Summary:</h4>

    <blockquote class="summary">
        <ul>
        </ul>
    </blockquote>

    <h4>Code:</h4>
    <blockquote class="code">
        <pre><code>
    </code></pre>
    </blockquote>

    <h4>Output:</h4>
    <blockquote class="output">
        <div>
            <?php
            ?>
        </div>
    </blockquote>
    <div class="footer">
        <a href="/Day/Day 0.php"> Day 0</a>
        | <a href="/">Home</a> |
        <a href="/Day/Day 0.php"> Day 0</a>
    </div>
</body>

</html>
```

# main.css
```css
@import url("https://fonts.googleapis.com/css2?family=Comfortaa&family=Fira+Code&family=Orbitron&family=Poppins&display=swap");

body {
  background-color: #e0e0e0;
  margin: 5em;
}
h1 {
  font-family: "Orbitron", sans-serif;
  font-size: 2em;
}
h2 {
  font-family: "Comfortaa", sans-serif;
  font-size: 1.5em;
}
h3 {
  font-family: "Comfortaa", sans-serif;
  font-size: 1.3em;
}
h4 {
  font-family: "Comfortaa", sans-serif;
  font-size: 1em;
}

blockquote {
  border-radius: 1em;
  background: #e0e0e0;
  box-shadow: 25px 25px 50px #cecece, -25px -25px 50px #f2f2f2;
  padding: 0.5em;
  margin-bottom: 2em;
}

blockquote a {
  text-decoration: none;
}

.summary {
  font-family: "Poppins", sans-serif !important;
  font-size: 1em;
}

.code {
  font-family: "Fira Code", monospace !important;
  font-size: 1em;
}

.output div {
  margin-left: 2em;
  margin-right: 2em;
  margin-top: 1em;
  margin-bottom: 1em;
  font-family: "Poppins", sans-serif !important;
  font-size: 1em;
}

blockquote ul {
    margin-left: 2em;
    margin-right: 2em;
    margin-top: 1em;
    margin-bottom: 1em;
    font-family: "Poppins", sans-serif !important;
    font-size: 1em;
  }

.footer {
  font-family: "Comfortaa", sans-serif;
  font-size: 1em;
}

.footer a {
  color: black;
  text-decoration: none;
}
```
# index.php
```php
<html>

<head>
    <title>PHP Tutorial</title>
    <link rel="icon" type="image/svg+xml" sizes="any" href="https://www.php.net/favicon.svg?v=2">
    <link rel="icon" type="image/png" sizes="196x196" href="https://www.php.net/favicon-196x196.png?v=2">
    <link rel="icon" type="image/png" sizes="32x32" href="https://www.php.net/favicon-32x32.png?v=2">
    <link rel="icon" type="image/png" sizes="16x16" href="https://www.php.net/favicon-16x16.png?v=2">
    <link rel="shortcut icon" href="https://www.php.net/favicon.ico?v=2">
</head>
<style>
    @import url("https://fonts.googleapis.com/css2?family=Comfortaa&family=Fira+Code&family=Orbitron&family=Poppins&display=swap");

    body {
        background-color: #e0e0e0;
        margin: 5em;
    }

    h1 {
        font-family: "Orbitron", sans-serif;
        font-size: 2em;
    }

    a {
        font-family: "Comfortaa", sans-serif;
        color: black;
        text-decoration: none;
        font-size: 2em;
        padding: 0.5em;
    }

    p {
        font-family: "Comfortaa", sans-serif;
        font-size: 1.5em;
    }

    li {
        padding: 0.5em;
    }

    ul {
        padding: 1em;
    }
</style>

<body>
    <h1>PHP Tutorial | Lessons Dashboard</h1>
    <p>In this tutorial, you will learn the basics of PHP, specifically tailored for Sri Lankan A/L students. Click on
        the lesson you want to continue.</p>
    <ul>
        <li><a href='/Day/Day 01.html'>Day 01.html</a></li>
        <?php
        // Specify the path to the "Day" folder
        $folderPath = "Day/";

        // Open the directory
        $dir = opendir($folderPath);

        // Check if the directory exists
        if ($dir === false) {
            echo "Error: Directory not found.";
            exit;
        }

        // Store the PHP files in an array for sorting
        $phpFiles = [];
        while (($file = readdir($dir)) !== false) {
            if (strpos($file, ".php") !== false) {
                $phpFiles[] = $file;
            }
        }

        // Sort the PHP files numerically
        natsort($phpFiles);

        // Create the links for the sorted PHP files
        foreach ($phpFiles as $file) {
            $link = $folderPath . $file;
            echo "<li><a href='$link'>$file</a></li>";
        }

        // Close the directory
        closedir($dir);
        ?>
    </ul>

</body>

</html>
```
