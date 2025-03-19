#------------------------------------------------------------------------------
# Query database for Foam Log extraction.
# Each line stands for one query:
#
#    name '/' line selector '/' column selector
#
# e.g.
#    kMin/bounding k,/min:
#
# The extraction will select a line using 'bounding k,' and in the line
# takes the word following 'min:' as the value.
#
# A special entry is the 'Separator' entry which delimits one iteration
# from the other.
#
#------------------------------------------------------------------------------

#- String to recognize new iteration by (usually same as 'Time')
# Separator/^[ \t]*pseudoTimeIteration = /pseudoTimeIteration = 
Separator/^[ \t]*pseudoTime: iteration /pseudoTime: iteration 

#- Time value:
# Time/^[ \t]*pseudoTimeIteration = /pseudoTimeIteration = 
Time/^[ \t]*pseudoTime: iteration /pseudoTime: iteration 

#- Res:
resRho/GMRES iteration: 0   Residual: /GMRES iteration: 0   Residual:
resUx/GMRES iteration: 0   Residual: /(
resE/GMRES iteration: 0   Residual: /)

