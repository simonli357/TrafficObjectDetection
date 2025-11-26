pipeline {
    agent none // Don't run on the main Jenkins node
    
    stages {
        stage('Test in Docker') {
            agent {
                docker { 
                    image 'python:3.9-alpine' 
                    // This creates the container and MOUNTS the current repo into it
                }
            }
            steps {
                // Jenkins has already cloned your repo into the workspace!
                // We just need to verify the file is there and run it.
                sh 'ls -la' 
                sh 'python test.py'
            }
        }
    }
}