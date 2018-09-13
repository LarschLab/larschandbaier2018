globals [ IADl IAD]


breed [type1s t1]
breed [type2s t2]

patches-own [
  deviceEdge                ;; true on nest patches, false elsewhere
]


turtles-own [
  boutR
  social
  ]


to setup

  clear-all
  create-type1s 1

  [ set color red
    set size 5
    setxy random-xcor / 2 random-ycor / 2
    set heading random 360
    set social t1-Social
    set boutR boutF
     ]

  create-type2s 1
  ; create one turtle
  [ set color yellow
    set size 5
    setxy random-xcor / 2 random-ycor / 2
    set heading random 360
    set social t2-Social
    set boutR boutF
     ]
  set IADl [0]
set IADl lput ( [distance turtle 1] of turtle 0) IADl
    setup-patches
    reset-ticks
end


to setup-patches
  ask patches
  [
    setup-deviceEdge ]
end

to setup-deviceEdge
  set deviceEdge (max-pxcor - 1) < sqrt(pycor ^ 2 + pxcor ^ 2) ;;or pycor <= min-pycor
  if deviceEdge
  [ set pcolor violet ]
end

to go
  ask type1s [
    set social t1-Social]

  ask type2s [
    set social t2-Social]

  ask turtles [ flock ]
  ;;repeat 5 [ ask turtles [ fd 0.2 ] display ] ;; smooth display
     ask turtles [ fd 0.2 ] ;; fast display
  tick
end

to flock  ;; turtle procedure

  turn-from-edges


  ifelse boutR = 0
    [set IAD cumul-av
    ifelse random-float 1 < social
      [cohere]
      [rt random 360]

      set boutR boutF]
  [set boutR boutR - 1]

end

to turn-from-edges ;; turtle procedure
    if[deviceEdge] of patch-ahead 1 = True
   [ rt (95 + random-float 90) ]

  if[deviceEdge] of patch-ahead 1 = True
   [ lt (185 + random-float 90) ]

end



;;; COHERE

to cohere  ;; turtle procedure

  let newHeading average-heading-towards-others + (random bout-error) - (bout-error / 2)
  let turn subtract-headings newHeading heading
  rt turn

end

to-report average-heading-towards-others  ;; turtle procedure
  ;; "towards myself" gives us the heading from the other turtle
  ;; to me, but we want the heading from me to the other turtle,
  ;; so we add 180
  let x-component mean [sin (towards myself + 180)] of other turtles
  let y-component mean [cos (towards myself + 180)] of other turtles
  ifelse x-component = 0 and y-component = 0
    [ report heading ]
    [ report atan x-component y-component ]
end

;;; HELPER PROCEDURES

to-report cumul-av
;; drop the first member of the list, but not until there are at least 1800 items in the list = 30 minutes
if (length IADl > 1800) [ set IADl but-first IADl ]
;; add the number of raindrops created in last tick to the end of the list
set IADl lput ( [distance turtle 1] of turtle 0) IADl
report mean IADl
end
@#$#@#$#@
GRAPHICS-WINDOW
15
13
629
628
-1
-1
6.0
1
10
1
1
1
0
0
0
1
-50
50
-50
50
0
0
1
ticks
30.0

BUTTON
686
46
749
79
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
693
103
756
136
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
911
158
1083
191
t1-Social
t1-Social
0
1
0.1726
.05
1
NIL
HORIZONTAL

SLIDER
1115
160
1287
193
t2-Social
t2-Social
0
1
0.1726
.05
1
NIL
HORIZONTAL

PLOT
911
208
1288
402
plot 1
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot (IAD)"

SLIDER
680
252
852
285
boutF
boutF
0
100
30.0
1
1
NIL
HORIZONTAL

SLIDER
680
214
852
247
turnR
turnR
0
1
0.0
.05
1
NIL
HORIZONTAL

SLIDER
683
292
855
325
bout-error
bout-error
0
90
0.0
5
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.0.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="vary t1 t2 social" repetitions="5" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="9000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-cohere-turn">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-separate-turn">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="vision">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.8"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minimum-separation">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.8"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="60"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="manyShort" repetitions="50" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="5000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-cohere-turn">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-separate-turn">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="vision">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minimum-separation">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
      <value value="0.8"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="60"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="manyShort" repetitions="100" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="5000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-cohere-turn">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-separate-turn">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="vision">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minimum-separation">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="60"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="manyShort_c" repetitions="120" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="5000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-cohere-turn">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-separate-turn">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="vision">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.15"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minimum-separation">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.15"/>
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="60"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="ManyParams" repetitions="50" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="5000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-cohere-turn">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-separate-turn">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="vision">
      <value value="150"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minimum-separation">
      <value value="0"/>
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.3"/>
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="60"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun_FineGrain" repetitions="4" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="72000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="7.0E-4"/>
      <value value="0.0015"/>
      <value value="0.0024"/>
      <value value="0.0034"/>
      <value value="0.0045"/>
      <value value="0.0057"/>
      <value value="0.0071"/>
      <value value="0.0086"/>
      <value value="0.0102"/>
      <value value="0.012"/>
      <value value="0.0141"/>
      <value value="0.0163"/>
      <value value="0.0188"/>
      <value value="0.0215"/>
      <value value="0.0246"/>
      <value value="0.0279"/>
      <value value="0.0317"/>
      <value value="0.0358"/>
      <value value="0.0404"/>
      <value value="0.0454"/>
      <value value="0.051"/>
      <value value="0.0573"/>
      <value value="0.0641"/>
      <value value="0.0717"/>
      <value value="0.0802"/>
      <value value="0.0895"/>
      <value value="0.0999"/>
      <value value="0.1113"/>
      <value value="0.124"/>
      <value value="0.1381"/>
      <value value="0.1536"/>
      <value value="0.1709"/>
      <value value="0.19"/>
      <value value="0.2111"/>
      <value value="0.2345"/>
      <value value="0.2604"/>
      <value value="0.2891"/>
      <value value="0.3209"/>
      <value value="0.3561"/>
      <value value="0.3951"/>
      <value value="0.4383"/>
      <value value="0.4861"/>
      <value value="0.539"/>
      <value value="0.5977"/>
      <value value="0.6626"/>
      <value value="0.7345"/>
      <value value="0.8141"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="7.0E-4"/>
      <value value="0.0015"/>
      <value value="0.0024"/>
      <value value="0.0034"/>
      <value value="0.0045"/>
      <value value="0.0057"/>
      <value value="0.0071"/>
      <value value="0.0086"/>
      <value value="0.0102"/>
      <value value="0.012"/>
      <value value="0.0141"/>
      <value value="0.0163"/>
      <value value="0.0188"/>
      <value value="0.0215"/>
      <value value="0.0246"/>
      <value value="0.0279"/>
      <value value="0.0317"/>
      <value value="0.0358"/>
      <value value="0.0404"/>
      <value value="0.0454"/>
      <value value="0.051"/>
      <value value="0.0573"/>
      <value value="0.0641"/>
      <value value="0.0717"/>
      <value value="0.0802"/>
      <value value="0.0895"/>
      <value value="0.0999"/>
      <value value="0.1113"/>
      <value value="0.124"/>
      <value value="0.1381"/>
      <value value="0.1536"/>
      <value value="0.1709"/>
      <value value="0.19"/>
      <value value="0.2111"/>
      <value value="0.2345"/>
      <value value="0.2604"/>
      <value value="0.2891"/>
      <value value="0.3209"/>
      <value value="0.3561"/>
      <value value="0.3951"/>
      <value value="0.4383"/>
      <value value="0.4861"/>
      <value value="0.539"/>
      <value value="0.5977"/>
      <value value="0.6626"/>
      <value value="0.7345"/>
      <value value="0.8141"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun" repetitions="120" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="72000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.15"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.15"/>
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun_FineGrainAll" repetitions="3" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="72000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <steppedValueSet variable="t2-Social" first="0" step="0.01" last="0.8"/>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <steppedValueSet variable="t1-Social" first="0" step="0.01" last="0.8"/>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun_FineGrainLots" repetitions="12" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="72000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0.0037"/>
      <value value="0.0066"/>
      <value value="0.0094"/>
      <value value="0.0124"/>
      <value value="0.0153"/>
      <value value="0.0183"/>
      <value value="0.0213"/>
      <value value="0.0243"/>
      <value value="0.0274"/>
      <value value="0.0305"/>
      <value value="0.0337"/>
      <value value="0.0369"/>
      <value value="0.0402"/>
      <value value="0.0435"/>
      <value value="0.0468"/>
      <value value="0.0502"/>
      <value value="0.0537"/>
      <value value="0.0572"/>
      <value value="0.0607"/>
      <value value="0.0644"/>
      <value value="0.068"/>
      <value value="0.0717"/>
      <value value="0.0755"/>
      <value value="0.0793"/>
      <value value="0.0832"/>
      <value value="0.0872"/>
      <value value="0.0912"/>
      <value value="0.0953"/>
      <value value="0.0995"/>
      <value value="0.1038"/>
      <value value="0.1081"/>
      <value value="0.1125"/>
      <value value="0.117"/>
      <value value="0.1216"/>
      <value value="0.1262"/>
      <value value="0.131"/>
      <value value="0.1358"/>
      <value value="0.1408"/>
      <value value="0.1458"/>
      <value value="0.151"/>
      <value value="0.1563"/>
      <value value="0.1617"/>
      <value value="0.1672"/>
      <value value="0.1729"/>
      <value value="0.1787"/>
      <value value="0.1846"/>
      <value value="0.1907"/>
      <value value="0.197"/>
      <value value="0.2034"/>
      <value value="0.21"/>
      <value value="0.2168"/>
      <value value="0.2238"/>
      <value value="0.231"/>
      <value value="0.2385"/>
      <value value="0.2461"/>
      <value value="0.254"/>
      <value value="0.2622"/>
      <value value="0.2707"/>
      <value value="0.2795"/>
      <value value="0.2887"/>
      <value value="0.2982"/>
      <value value="0.3081"/>
      <value value="0.3184"/>
      <value value="0.3292"/>
      <value value="0.3405"/>
      <value value="0.3524"/>
      <value value="0.3649"/>
      <value value="0.3781"/>
      <value value="0.3921"/>
      <value value="0.4069"/>
      <value value="0.4227"/>
      <value value="0.4397"/>
      <value value="0.4579"/>
      <value value="0.4777"/>
      <value value="0.4993"/>
      <value value="0.523"/>
      <value value="0.5493"/>
      <value value="0.5789"/>
      <value value="0.6128"/>
      <value value="0.6522"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0.0037"/>
      <value value="0.0066"/>
      <value value="0.0094"/>
      <value value="0.0124"/>
      <value value="0.0153"/>
      <value value="0.0183"/>
      <value value="0.0213"/>
      <value value="0.0243"/>
      <value value="0.0274"/>
      <value value="0.0305"/>
      <value value="0.0337"/>
      <value value="0.0369"/>
      <value value="0.0402"/>
      <value value="0.0435"/>
      <value value="0.0468"/>
      <value value="0.0502"/>
      <value value="0.0537"/>
      <value value="0.0572"/>
      <value value="0.0607"/>
      <value value="0.0644"/>
      <value value="0.068"/>
      <value value="0.0717"/>
      <value value="0.0755"/>
      <value value="0.0793"/>
      <value value="0.0832"/>
      <value value="0.0872"/>
      <value value="0.0912"/>
      <value value="0.0953"/>
      <value value="0.0995"/>
      <value value="0.1038"/>
      <value value="0.1081"/>
      <value value="0.1125"/>
      <value value="0.117"/>
      <value value="0.1216"/>
      <value value="0.1262"/>
      <value value="0.131"/>
      <value value="0.1358"/>
      <value value="0.1408"/>
      <value value="0.1458"/>
      <value value="0.151"/>
      <value value="0.1563"/>
      <value value="0.1617"/>
      <value value="0.1672"/>
      <value value="0.1729"/>
      <value value="0.1787"/>
      <value value="0.1846"/>
      <value value="0.1907"/>
      <value value="0.197"/>
      <value value="0.2034"/>
      <value value="0.21"/>
      <value value="0.2168"/>
      <value value="0.2238"/>
      <value value="0.231"/>
      <value value="0.2385"/>
      <value value="0.2461"/>
      <value value="0.254"/>
      <value value="0.2622"/>
      <value value="0.2707"/>
      <value value="0.2795"/>
      <value value="0.2887"/>
      <value value="0.2982"/>
      <value value="0.3081"/>
      <value value="0.3184"/>
      <value value="0.3292"/>
      <value value="0.3405"/>
      <value value="0.3524"/>
      <value value="0.3649"/>
      <value value="0.3781"/>
      <value value="0.3921"/>
      <value value="0.4069"/>
      <value value="0.4227"/>
      <value value="0.4397"/>
      <value value="0.4579"/>
      <value value="0.4777"/>
      <value value="0.4993"/>
      <value value="0.523"/>
      <value value="0.5493"/>
      <value value="0.5789"/>
      <value value="0.6128"/>
      <value value="0.6522"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun_FineGrainRealPs" repetitions="12" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="72000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0.0099"/>
      <value value="0.0143"/>
      <value value="0.0254"/>
      <value value="0.0271"/>
      <value value="0.0466"/>
      <value value="0.0607"/>
      <value value="0.0622"/>
      <value value="0.0746"/>
      <value value="0.0817"/>
      <value value="0.0954"/>
      <value value="0.0978"/>
      <value value="0.1289"/>
      <value value="0.1297"/>
      <value value="0.1501"/>
      <value value="0.2026"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0.0099"/>
      <value value="0.0143"/>
      <value value="0.0254"/>
      <value value="0.0271"/>
      <value value="0.0466"/>
      <value value="0.0607"/>
      <value value="0.0622"/>
      <value value="0.0746"/>
      <value value="0.0817"/>
      <value value="0.0954"/>
      <value value="0.0978"/>
      <value value="0.1289"/>
      <value value="0.1297"/>
      <value value="0.1501"/>
      <value value="0.2026"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="paperRun_FineGrainRealPs_review" repetitions="6" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="9000"/>
    <metric>IAD</metric>
    <enumeratedValueSet variable="boutF">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t2-Social">
      <value value="0"/>
      <value value="0.0067"/>
      <value value="0.0085"/>
      <value value="0.0161"/>
      <value value="0.0371"/>
      <value value="0.0391"/>
      <value value="0.045"/>
      <value value="0.0555"/>
      <value value="0.071"/>
      <value value="0.0762"/>
      <value value="0.0838"/>
      <value value="0.085"/>
      <value value="0.097"/>
      <value value="0.1212"/>
      <value value="0.1726"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="turnR">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="t1-Social">
      <value value="0"/>
      <value value="0.0067"/>
      <value value="0.0085"/>
      <value value="0.0161"/>
      <value value="0.0371"/>
      <value value="0.0391"/>
      <value value="0.045"/>
      <value value="0.0555"/>
      <value value="0.071"/>
      <value value="0.0762"/>
      <value value="0.0838"/>
      <value value="0.085"/>
      <value value="0.097"/>
      <value value="0.1212"/>
      <value value="0.1726"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bout-error">
      <value value="0"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
