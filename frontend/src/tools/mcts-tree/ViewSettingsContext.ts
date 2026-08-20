import { createContext, useContext } from 'react'

/** Filter/display settings that apply uniformly to every row no matter how
 * deep in the tree it is — carried via context instead of threading two more
 * props through every recursive call. (`compareNode` and `expandedPaths`
 * stay as explicit props on `NodeBlock`/edge rows: those genuinely change
 * per position in the tree, so they're not "settings".) */
export interface ViewSettings {
  filterText: string
  hideZero: boolean
}

export const ViewSettingsContext = createContext<ViewSettings>({ filterText: '', hideZero: false })

export function useViewSettings(): ViewSettings {
  return useContext(ViewSettingsContext)
}
