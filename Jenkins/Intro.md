```whatever
 Jenkins is
1) Open source automation server
2) Helps to automate the non-human part of
the software development process
3) Allows continuous integration
5) In my company Port is 8081 localhost
 To run my project in Jenkins
1) I need  to Login Jenkins account
2) I need to Create freestyle project
3) after that Install plugins -cucumber report and git
4) Under source code management I choose git and paste git url
5) I Build trigger and I choose to build periodically
6) Invoke top-level maven
2. Goals; clean verify -
Drunner=smoke_runner.xml
7) Under post-build actions
1. Choose cucumber reports
2. Choose editable email notify
8) Editable email notification
1. Attach build log; choose build log
2. Click advanced settings
9) Failure-Any
1. Click advanced
2. Recipient list - email address who will
receive the report. Add comma if multiple
3. Click add trigger - like failure always
4. Attach build log; select attach build log
5. Save
10) Final Step
1. Click build now and test will run and
gives your cucumber report
```
```whatever
Continuous Integration and Deployment tool. 3 components ofJenkins
○ 1. Code change à Devs makes changes to the applicationcode
○ 2. Test à CI tool automatically picks up the changes and tests the application
○ 3. Deploy à CI tool deploys the application with changes
```
https://www.tutorialspoint.com/jenkins/index.htm