{-# LANGUAGE OverloadedLists#-}
{-# LANGUAGE RankNTypes#-}
{-# LANGUAGE Strict#-}
module Lib
    ( someFunc
    , acFunc
    , applyNeuron
    , notN
    , train
    , perceptron
    , trainSet
    , gradDesc
    , funkcja1
    , funkcja2
    ) where

import Control.Monad.ST
import Debug.Trace
import qualified Numeric.AD as AD
import Numeric.AD.Internal.Reverse
import Numeric.LinearAlgebra
import qualified Data.Vector.Storable as VS

someFunc :: IO ()
someFunc = putStrLn "someFunc"

data Neuron = Neuron {
                     w  :: Vector Double,
                     f  :: Double -> Double }

data MNeuron = MNeuron {
                       mw :: Matrix Double,
                       mf :: Double -> Double
                       }

instance Show Neuron where
  show = show . w

applyMNeuron :: MNeuron -> Vector Double -> Vector Double
applyMNeuron (MNeuron wagi funkcja) = VS.map funkcja . (wagi #>)

applyNeuron :: Neuron -> Vector Double -> Double
applyNeuron (Neuron w f) = f . (w <.>) . VS.cons 1

acFunc :: Double -> Double
acFunc x | x >= 0 = 1
          | otherwise = 0

notN = Neuron [1,-1.5] acFunc

andN = Neuron [-2, 1.5, 1.5] acFunc

nandN = Neuron [2, -1.5, -1.5] acFunc

orN = Neuron [-1, 1.5, 1.5] acFunc

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

prPrint :: Vector Double -> String
prPrint = morph (\x (acc, count) -> ((if x == -1 then " " else "*") ++ (if count `mod` 5 == 0 then "\n" else []) ++ acc, count+1)) []
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
    w1 = map (/25) . VS.foldr (\i acc -> VS.foldr (\j ac -> (i * j) : ac) acc v1) [] $ v1
    w2 = map (/25) . VS.foldr (\i acc -> VS.foldr (\j ac -> (i * j) : ac) acc v2) [] $ v2
    weights = (25><25) $ zipWith (+) w1 w2

pics2 :: [String]
pics2 = [ " *** \n * * \n * * \n *** \n     "
        , "  *  \n  *  \n  *  \n  *  \n  *  "]

gradStep :: Floating a => a
gradStep = 0.01

eps :: Floating a => a
eps = 0.00001

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
