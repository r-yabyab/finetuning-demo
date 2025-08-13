sudo apt install openjdk-21-jdk

# Download JUnit 5 platform and API jars, need to point to this to run junit tests
# Example locations, adjust paths
wget https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar

javac -cp .:junit-platform-console-standalone-1.10.0.jar Calculator.java CalculatorTest.java
java -jar junit-platform-console-standalone-1.10.0.jar -cp . --scan-classpath