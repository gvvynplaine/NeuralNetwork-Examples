mkdir -p classes

javac -J-Xms512m -J-Xmx512m -cp target/deeplearning4j-0.4-rc3.8.jar -d classes `find src -type f -name "*.java"`

