'use client'

/**
 * Copied from supabase/supabase ui package.
 */

import { useState, useEffect } from 'react'
import styles from './ai-icon-animation-style.module.css'
import { cn } from '~/lib/utils'

interface Props {
  loading?: boolean
  className?: string
  allowHoverEffect?: boolean
}

const AiIconAnimation = ({ loading = false, className, allowHoverEffect = false }: Props) => {
  const [step, setStep] = useState(1)
  const [exitStep, setExitStep] = useState(1)
  const [isAnimating, setIsAnimating] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      if (loading) {
        setIsAnimating(true)
        setStep((step) => {
          return (step % 5) + 1
        })
        setExitStep((step) => {
          return (step % 5) + 1
        })
      }
    }, 500)
    return () => clearInterval(interval)
  }, [loading])

  useEffect(() => {
    if (loading === false) {
      setTimeout(() => {
        setIsAnimating(false)
      }, 500)
      setStep(1)
    }
  }, [loading])

  return (
    <div
      className={cn(
        styles['ai-icon__container'],
        allowHoverEffect && styles['ai-icon__container--allow-hover-effect'],
        loading && isAnimating && styles['spin-ai-icon-container'],
        className
      )}
    >
      <div className={cn(styles['ai-icon__grid'])}>
        <div
          className={cn(
            styles['ai-icon__grid__square'],
            loading
              ? styles[`ai-icon__grid__square--step-${step}`]
              : isAnimating
                ? styles[`ai-icon__grid__square--exiting--step-${exitStep}`]
                : styles[`ai-icon__grid__square--static`]
          )}
        ></div>
        <div
          className={cn(
            styles['ai-icon__grid__square'],
            loading
              ? styles[`ai-icon__grid__square--step-${step}`]
              : isAnimating
                ? styles[`ai-icon__grid__square--exiting--step-${exitStep}`]
                : styles[`ai-icon__grid__square--static`]
          )}
        ></div>
        <div
          className={cn(
            styles['ai-icon__grid__square'],
            loading
              ? styles[`ai-icon__grid__square--step-${step}`]
              : isAnimating
                ? styles[`ai-icon__grid__square--exiting--step-${exitStep}`]
                : styles[`ai-icon__grid__square--static`]
          )}
        ></div>
        <div
          className={cn(
            styles['ai-icon__grid__square'],
            loading
              ? styles[`ai-icon__grid__square--step-${step}`]
              : isAnimating
                ? styles[`ai-icon__grid__square--exiting--step-${exitStep}`]
                : styles[`ai-icon__grid__square--static`]
          )}
        ></div>
      </div>
    </div>
  )
}

export { AiIconAnimation }
