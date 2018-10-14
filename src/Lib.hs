{-# LANGUAGE OverloadedLists#-}
{-# LANGUAGE RankNTypes#-}
{-# LANGUAGE ScopedTypeVariables #-}
--{-# LANGUAGE Strict#-}
module Lib where

import Control.Monad.ST
import Debug.Trace
import qualified Numeric.AD as AD
import Numeric.AD.Internal.Reverse
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import qualified Data.Vector.Storable as VS
import System.Random
import Debug.Trace
import Text.Printf
import Control.Monad
import Control.Concurrent
import Data.List (permutations)

someFunc :: IO ()
someFunc = putStrLn "someFunc"

data Neuron = Neuron {
                     w  :: Vector Double,
                     f  :: Double -> Double }

data MNeuron = MNeuron {
                       mw :: Matrix Double,
                       mf :: Double -> Double }

instance Show Neuron where
  show = show . w

instance Show MNeuron where
  show = show . mw

applyMNeuron :: MNeuron -> Vector Double -> Vector Double
applyMNeuron (MNeuron wagi funkcja) = VS.map funkcja . (wagi #>) . VS.cons 1

applyNeuron :: Neuron -> Vector Double -> Double
applyNeuron (Neuron w f) = f . (w <.>) . VS.cons 1

acFunc :: Double -> Double
acFunc x | x >= 0 = 1
          | otherwise = 0

notN = Neuron [1,-1.5] acFunc

andN = Neuron [-2, 1.5, 1.5] acFunc

nandN = Neuron [2, -1.5, -1.5] acFunc

orN = Neuron [-1, 1.5, 1.5] acFunc

-- 2. Picture classification

trainSet :: [(Vector Double, Double)]
trainSet = [([ 0, 0, 0, 0, 0
              , 0, 1, 1, 0, 0
              , 0, 0, 1, 0, 0
              , 0, 0, 1, 0, 0
              , 0, 0, 1, 0, 0], 1)

           , ([ 0, 0, 1, 1, 0
              , 0, 0, 0, 1, 0
              , 0, 0, 0, 1, 0
              , 0, 0, 0, 0, 0
              , 0, 0, 0, 0, 0], 1)

           , ([ 0, 0, 0, 0, 0
              , 1, 1, 0, 0, 0
              , 0, 1, 0, 0, 0
              , 0, 1, 0, 0, 0
              , 0, 1, 0, 0, 0], 1)

           , ([ 0, 0, 0, 0, 0
              , 0, 1, 1, 1, 0
              , 0, 1, 0, 1, 0
              , 0, 1, 1, 1, 0
              , 0, 0, 0, 0, 0], 0)

           , ([ 0, 0, 0, 0, 0
              , 0, 0, 0, 0, 0
              , 1, 1, 1, 0, 0
              , 1, 0, 1, 0, 0
              , 1, 1, 1, 0, 0], 0)]

test :: [Vector Double]
test = [[ 0, 1, 1, 0, 0
        , 0, 0, 1, 0, 0
        , 0, 0, 1, 0, 0
        , 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0
        ]
       ,[ 1, 1, 1, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 1, 0, 0
        , 0, 0, 0, 0, 0
        ]
       ,[ 0, 0, 1, 0, 0
        , 0, 0, 1, 0, 0
        , 0, 0, 1, 0, 0
        , 0, 0, 1, 0, 0
        , 0, 0, 0, 0, 0
        ]
       ,[ 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0
        , 1, 1, 1, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 1, 0, 0
        ]]

train st perc trSet = go st 0 perc (cycle trSet) (length trSet)
  where
    go st iters n@(Neuron w f) ((inp,z):xs) c =
      if c <=0 then
        (iters, n)
      else
        go st (iters+1) (Neuron newW f) xs newC
      where
        y = applyNeuron n inp
        newC = if z == y then c - 1 else 5
        newW = w + VS.map (\x -> x * st * (z - y)) (VS.cons 1 inp)

perceptron :: Neuron
perceptron = Neuron (VS.fromList . replicate 26 $ 1) acFunc

-- 3. Picture recognition

prPrint :: Vector Double -> String
prPrint = morph (\x (acc, count) -> ((if x <= 0 then " " else "■") ++ (if count `mod` 5 == 0 then "\n" else []) ++ acc, count+1)) []
  where
    morph f a = fst . VS.foldr f (a,0)

readPic :: String -> Vector Double
readPic = VS.fromList . foldr (\x acc -> case x of
                                  ' ' -> (-1):acc
                                  '*' -> 1:acc
                                  '\n'-> acc) []
pics :: [String]
pics = [ "     \n *** \n * * \n *** \n     "
       , "     \n **  \n  *  \n  *  \n     "]

sgn :: Double -> Double
sgn x
     | x >= 0 = 1
     | x < 0 = -1

picTaker :: MNeuron
picTaker = MNeuron weights sgn
  where
    [v1, v2] = map readPic pics
    w1       = cmap (/25) (v1 `outer` v1)
    w2       = cmap (/25) (v2 `outer` v2)
    weights = w1 + w2

apMNeu2 (MNeuron weights f) = VS.map f . (weights #>)

pics2 :: [String]
pics2 = [ " *** \n * * \n * * \n *** \n     "
        , "  *  \n  *  \n  *  \n  *  \n  *  "]

-- 4. Gradient Descent
gradStep :: Floating a => a
gradStep = 0.1

eps :: Floating a => a
eps = 0.000001

gradDesc :: (Floating a, Ord a, Show a) => (forall b. (Floating b, Ord b, Show b) => [b] -> b) -> [a] -> ([a], a)
gradDesc f x = if diff < eps then (newX, f newX) else gradDesc f newX
  where gr   = AD.grad f x
        newX = zipWith (-) x . map (gradStep *) $ gr
        diff = abs . foldr (+) 0 . zipWith (-) x $ newX

funkcja1 :: Num a => [a] -> a
funkcja1 [x,y,z] = (2 * x * x) + (2 * y * y) + (z * z) - (2 * x * y) - (2 * y * z) - (2 * x) + 3

dfunkcja1 :: Num a => [a] -> [a]
dfunkcja1 [x,y,z] = [(4*x) - (2*y) - 2, 4*y - 2*x - 2*z, 2*z - 2*y]

funkcja2 :: Num a => [a] -> a
funkcja2 [x,y] = (3 * x * x * x * x) + (4 * x * x* x) - (12 * x * x) + (12 * y * y) - (27 * y)

dfunkcja2 :: Num a => [a] -> [a]
dfunkcja2 [x,y] = [12* x* x* x + 12* x* x - 24* x, 24 * y - 24]

-- 5. Backpropagation

beta :: Double
beta = 1.25

krok :: Double
krok = 1

sig :: Double -> Double
sig = (1/).(1+).exp.negate

dsig :: Double -> Double
dsig x = sig x * (1 - sig x)

trainSetXOR :: [(Vector Double, Double)]
trainSetXOR =
  [([0,0],0)
  ,([1,0],1)
  ,([0,1],1)
  ,([1,1],0)]

fstLayer :: IO (Neuron, Neuron)
fstLayer = do
  randVec1 <- VS.replicateM 3 randomIO
  randVec2 <- VS.replicateM 3 randomIO
  return (Neuron randVec1 sig, Neuron randVec2 sig)

outLayer :: IO Neuron
outLayer = do
  randVec <- VS.replicateM 3 randomIO
  return (Neuron randVec sig)

beginning :: ((Neuron, Neuron), Neuron)
beginning = ((Neuron [2,0,1] sig, Neuron [2,0,1] sig), Neuron [2,0,1] sig)

applyXor :: ((Neuron,Neuron), Neuron) -> Vector Double -> Double
applyXor ((h1, h2), o) u = applyNeuron o out
  where
    x1 = applyNeuron h1 u
    x2 = applyNeuron h2 u
    out= [x1, x2]

infixr 3 #*
(#*) :: Double -> Vector Double -> Vector Double
x #* v = VS.map (*x) v

zipWith4 :: (a -> b -> c -> d -> e) -> [a] -> [b] -> [c] -> [d] -> [e]
zipWith4 f x y z = zipWith ($) (zipWith3 f x y z)

trainXOR perc trSet = go 0 perc
  where
    go iters n@((n1@(Neuron w1 f1), n2@(Neuron w2 f2)), n3@(Neuron s f3)) =
      if eps > (maximum . VS.toList .  VS.map abs $ difference) then
        (iters, newN)
      else
--        (if iters `mod` 10000 == 0 then trace (show iters ++ " iterations. Gradients:\n" ++ unlines [show gradS,show gradW1,show gradW2] ++ "\nDifference:\n" ++ show difference ++ "\n") else id ) $
        go (iters+1) newN
      where
        difference = (w1 VS.++ w2 VS.++s) - (newW1 VS.++ newW2 VS.++ newS)
        newN =  ((Neuron newW1 f1, Neuron newW2 f2), Neuron newS f3)
        u   = map (VS.cons 1 . fst) trSet
        x1s = map (applyNeuron n1 . fst) trSet
        x2s = map (applyNeuron n2 . fst) trSet
        x :: [Vector Double]
        x   = zipWith (\x1 x2 -> [1,x1,x2]) x1s x2s
        y   = map (applyXor n . fst) trSet
        z   = map snd trSet
        grs x y z = dsig (x<.>s) * (y - z) #* x
        gradsS= zipWith3 grs x y z
        gradS = foldr (+) [0,0,0] gradsS
        newS = s - (VS.map (*gradStep) gradS)
        [_,s1,s2] = s
        grw w sx x y z u= (y-z) * sx * dsig (x<.>s) * dsig (w<.>u) #* u
        gradsW w s = zipWith4 (grw w s) x y z u
        gradW w s = foldr (+) [0,0,0] $ gradsW w s
        gradW1 = gradW w1 s1
        gradW2 = gradW w2 s2
        newW1 = w1 - (VS.map (*gradStep) $ gradW w1 s1)
        newW2 = w2 - (VS.map (*gradStep) $ gradW w2 s2)

prettyPrintVals :: [Double] -> String
prettyPrintVals = unwords . map (printf "%.4f")

-- Zad. 6 Encoder - Decoder

autoEncTrain = map readPic [ "     \n **  \n  *  \n  *  \n  *  "
                            , "  ** \n   * \n   * \n     \n     "
                            , "     \n**   \n *   \n *   \n *   "]

type EncDec = (MNeuron, MNeuron)

makeDec (_, MNeuron w _) = MNeuron w acFunc

zeroM inp out = (out><inp) $ replicate (out*inp) 0

--startEncDec :: IO EncDec
startEncDec = do --(MNeuron (zeroM 26 16) sig, MNeuron (zeroM 17 25) sig) --do
  m1 <- map (\x -> x - 0.5) <$> replicateM (25*26) (randomIO :: IO Double)
  m2 <- map (\x -> x - 0.5) <$> replicateM (25*17) (randomIO :: IO Double)
  return (MNeuron ((16><26) m1) sig, MNeuron ((25><17) m2) sig)

decode (_,a) = applyMNeuron a

encode (a,_) = applyMNeuron a

ende a = applyMNeuron (makeDec a) . encode a

alfa = 0.8

trainED trSet = trainHelp 0
  where trainHelp old ed@(MNeuron w f1, MNeuron w' f2) =
          if abs (old - rmse) < 0.000001 then ed
          else trace ("RMSE = "++ show rmse) $ trainHelp rmse (MNeuron nW f1, MNeuron nW' f2)
          where nW            = w - cmap (*alfa) gradW
                nW'           = w'- cmap (*alfa) gradW'
                diffDecode :: Vector Double -> Vector Double
                diffDecode    = cmap dsig . (#>) w' . VS.cons 1
                diffEncode    = cmap dsig . (#>) w  . VS.cons 1
                gradDec x y x'= let dx = (x' - x) * diffDecode y
                                in dx `outer` VS.cons 1 y
                gradEnc x y x'= let dx = (x' - x) * diffDecode y
                                    dy = VS.tail (tr w' #> dx) * diffEncode x
                                in dy `outer` VS.cons 1 x
                triples       = map (\x -> (x, encode ed x,decode ed . encode ed $ x )) trSet
                gradsW'       = map (\(x,y,x') -> gradDec x y x') triples
                gradsW        = map (\(x,y,x') -> gradEnc x y x') triples
                rmse          = sqrt . VS.foldr (+) 0 . foldr (\(x,_,x') acc -> (x-x')^2 +acc ) (fromList . replicate 25 $ 0) $ triples
                gradW'        = foldr (+) (zeroM 17 25) gradsW'
                gradW         = foldr (+) (zeroM 26 16) gradsW

-- Zad 7. Sieci Hopfielda

readPic2 :: String -> Vector Double
readPic2 = VS.fromList . foldr (\x acc -> case x of
                                  ' ' -> 0:acc
                                  '*' -> 1:acc
                                  '\n'-> acc) []

jedynka :: Vector Double
jedynka = readPic2 "     \n **  \n  *  \n  *  \n  *  "

wagi :: Matrix Double
wagi = cmap (*2) $ (xs `outer` xs) * (cmap ((+1).negate) $ ident 25)
  where xs = cmap (flip (-) 0.5) jedynka

bias :: Vector Double
bias = cmap (pred.(/2)) (wagi #> fromList (replicate 25 1))

iteration :: Vector Double -> Vector Double
iteration = (flip (-) bias).(wagi #>) >>= VS.zipWith (\u x -> if u > 0 then 1 else if u < 0 then 0 else x)

inp :: Vector Double
inp = readPic2 " *   \n   * \n* *  \n    *\n *   "

process = process . iteration . (prPrint >>= trace)

randInp :: IO (Vector Double)
randInp = VS.fromList . map (\x -> if x > 0.5 then 1 else 0) <$> replicateM 25 (randomIO :: IO Double)

zeroV :: Vector Double
zeroV = readPic2 " *** \n * * \n * * \n * * \n *** "

wagi2 :: Matrix Double
wagi2 = cmap (*2) (((cx `outer` cx) + (dx `outer` dx)) * (cmap ((+1).negate) $ ident 25))
  where dx = cmap (flip (-) 0.5) zeroV
        cx = cmap (flip (-) 0.5) jedynka

bias2 :: Vector Double
bias2 = cmap (pred.pred.(/2)) (wagi2 #> fromList (replicate 25 1))

iterate2 = (flip (-) bias2).(wagi2 #>) >>= VS.zipWith (\u x -> if u > 0 then 1 else if u < 0 then 0 else x)

inp2 :: Vector Double
inp2 = readPic2 "  *  \n * * \n     \n* *  \n    *"

-- Boltzmann

prob :: Double -> Double -> Double
prob t x = (1/).(1+).exp.negate $ (x/t)

boltzStep :: Double -> Vector Double -> IO (Vector Double)
boltzStep temp = VS.mapM (\x -> randomIO >>= \y -> if y <= prob temp x then return 1 else return 0).(flip (-) bias).(wagi#>)

--replicateIO :: (a -> IO a) -> a -> IO [a]
--replicateIO f a = [ a : rest | rest <- replicateIO f nA , nA <- f a ]

annealing :: Double -> Vector Double -> IO ()
annealing = aSt 0

aSt :: Double -> Double -> Vector Double -> IO ()
aSt x inT input = do
  nv <- boltzStep (tann inT x) input
  threadDelay 300000
  putStrLn $ "Step number: " ++ show x
  putStrLn $ "Temperature: " ++ show (tann inT x)
  putStrLn . prPrint $ input
  aSt (x+1) inT nv

bSt :: Double -> Vector Double -> IO ()
bSt temp input = do
  nv <- boltzStep temp input
  threadDelay 300000
  putStrLn . prPrint $ input
  bSt temp nv

tann :: Double -> Double -> Double
tann t = (t/).(1+).log.(1+)

-- TSP

randomPermutation :: Int -> IO [Int]
randomPermutation n = do
  let ps= permutations [0..n-1]
      x = length ps
  r <- randomRIO (0,x-1) :: IO Int
  return $ ps !! r

odleglosci1 = cmap abs (row [1..10] - col [1..10]) - (col (8:replicate 9 0) * row (replicate 9 0 ++ [1])) - (col (replicate 9 0++[8]) * row (1 : replicate 9 0))
